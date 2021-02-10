/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/job_rewriter/calculation_pass.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

// Do InsertNcclLogicalOpPass will use backward recomputation for sublinear memory cost.
class InsertNcclLogicalOpPass final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InsertNcclLogicalOpPass);
  InsertNcclLogicalOpPass() = default;
  ~InsertNcclLogicalOpPass() = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }

  bool IsEnabled(const JobPassCtx& ctx) const {
#if defined(WITH_CUDA) && NCCL_VERSION_CODE > 2700
    return Global<ResourceDesc, ForSession>::Get()->resource().enable_insert_nccl_logical_op_pass();
#else
    return false;
#endif
  }

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;
};

const std::string kNcclLogicalOpNamePrefix = "OneFlow-System-NCCL-logical-Op";
const std::string kNoneNcclOpTypeName = "DoNotInsertNcclLogialOp";

void FindMaxConnectedSubgraphForGpuExecOrder(HashSet<const OpNode*>* ret, const OpGraph& op_graph,
                                             const std::vector<const OpNode*>& order) {
  HashSet<const OpNode*> visited;

  for (const OpNode* seed_node : order) {
    if (visited.find(seed_node) != visited.end()) { continue; }
    CHECK(visited.insert(seed_node).second);
    const ParallelDesc& seed_parallel_desc = seed_node->parallel_desc();
    // NOTE(chengcheng): ONLY consider GPU op and parallel num > 1.
    if (seed_parallel_desc.device_type() != DeviceType::kGPU) { continue; }
    if (seed_parallel_desc.parallel_num() <= 1) { continue; }
    // NODE(chengcheng): Exclude op that change the time shape.
    //   like pack/unpack, repeat/acc, etc.
    if (!seed_node->IsTimeShapeIdentity()) { continue; }

    HashSet<const OpNode*> this_subgraph;
    std::queue<const OpNode*> queued_nodes;
    queued_nodes.push(seed_node);
    while (!queued_nodes.empty()) {
      const OpNode* cur_node = queued_nodes.front();
      queued_nodes.pop();

      CHECK(cur_node->parallel_desc() == seed_parallel_desc);
      CHECK(this_subgraph.insert(cur_node).second);

      cur_node->ForEachNodeOnInOutEdge([&](const OpNode* next_node) {
        if (visited.find(next_node) == visited.end()
            && next_node->parallel_desc() == seed_parallel_desc
            && next_node->IsTimeShapeIdentity()) {
          CHECK(visited.insert(next_node).second);
          queued_nodes.push(next_node);
        }
      });
    }

    if (this_subgraph.size() > ret->size()) { ret->swap(this_subgraph); }
  }
}

bool TryGetNcclLogicalOpConf(OperatorConf* ret, const OpNode* src_node, const OpNode* dst_node,
                             const LogicalBlobId& lbi) {
  const int64_t scope_symbol_id = src_node->op().op_conf().scope_symbol_id();
  const std::string lbn = GenLogicalBlobName(lbi);
  const SbpParallel& src_sbp = src_node->SbpParallel4Lbi(lbi);
  const SbpParallel& dst_sbp = dst_node->SbpParallel4Lbi(lbi);
  const BlobDesc& logical_blob_desc = src_node->LogicalBlobDesc4Lbi(lbi);
  const ParallelDesc& parallel_desc = src_node->parallel_desc();

  // NOTE(chengcheng): nccl donot support dynamic shape.
  if (logical_blob_desc.is_dynamic()) { return false; }
  CHECK_GT(logical_blob_desc.shape().elem_cnt(), 0);
  CHECK_GT(logical_blob_desc.shape().NumAxes(), 0);
  CHECK_GT(logical_blob_desc.shape().At(0), 0);
  if (src_sbp.has_partial_sum_parallel() && dst_sbp.has_broadcast_parallel()) {
    // P2B : AllReduce
    user_op::UserOpConfWrapper nccl_op_wrapper =
        user_op::UserOpConfWrapperBuilder(kNcclLogicalOpNamePrefix + "-P2B-" + NewUniqueId())
            .Op("_nccl_logical_op_all_reduce")
            .Input("in", lbn)
            .Output("out")
            .ScopeSymbolId(scope_symbol_id)
            .Build();
    *ret = nccl_op_wrapper.op_conf();
    std::cout << "cclog: insert nccl op: " << ret->name() << std::endl;
    return true;
  }
  if ((logical_blob_desc.shape().At(0) % parallel_desc.parallel_num() == 0)
      && (src_sbp.has_partial_sum_parallel() && dst_sbp.has_split_parallel())
      && (dst_sbp.split_parallel().axis() == 0)) {
    // P2S : ReduceScatter
    user_op::UserOpConfWrapper nccl_op_wrapper =
        user_op::UserOpConfWrapperBuilder(kNcclLogicalOpNamePrefix + "-P2S-" + NewUniqueId())
            .Op("_nccl_logical_op_reduce_scatter")
            .Input("in", lbn)
            .Output("out")
            .ScopeSymbolId(scope_symbol_id)
            .Build();
    *ret = nccl_op_wrapper.op_conf();
    std::cout << "cclog: insert nccl op: " << ret->name() << std::endl;
    return true;
  }
  if ((logical_blob_desc.shape().At(0) % parallel_desc.parallel_num() == 0)
      && (src_sbp.has_split_parallel() && dst_sbp.has_broadcast_parallel())
      && (src_sbp.split_parallel().axis() == 0)) {
    // S2B : AllGather
    user_op::UserOpConfWrapper nccl_op_wrapper =
        user_op::UserOpConfWrapperBuilder(kNcclLogicalOpNamePrefix + "-S2B-" + NewUniqueId())
            .Op("_nccl_logical_op_all_gather")
            .Input("in", lbn)
            .Output("out")
            .ScopeSymbolId(scope_symbol_id)
            .Build();
    *ret = nccl_op_wrapper.op_conf();
    std::cout << "cclog: insert nccl op: " << ret->name() << std::endl;
    return true;
  }
  if ((src_sbp.has_split_parallel() && dst_sbp.has_split_parallel())
      && (src_sbp.split_parallel().axis() != dst_sbp.split_parallel().axis())
      && (logical_blob_desc.shape().At(src_sbp.split_parallel().axis())
              % parallel_desc.parallel_num()
          == 0)
      && (logical_blob_desc.shape().At(dst_sbp.split_parallel().axis())
              % parallel_desc.parallel_num()
          == 0)) {
    // S2S : All2All
    /*
     * TODO(chengcheng)
    user_op::UserOpConfWrapper nccl_op_wrapper =
        user_op::UserOpConfWrapperBuilder(kNcclLogicalOpNamePrefix + "-S2S-" + NewUniqueId())
            .Op("_nccl_logical_op_all2all")
            .Input("in", lbn)
            .Output("out")
            .ScopeSymbolId(scope_symbol_id)
            .Build();
    *ret = nccl_op_wrapper.op_conf();
    */
    std::cout << "cc WARNING: Need insert nccl all2all op BUT UNIMPLEMENTED(). OpEdge : "
              << src_node->op().op_name() << " -> " << dst_node->op().op_name()
              << " And the logical shape elem cnt = : " << logical_blob_desc.shape().elem_cnt()
              << std::endl;
    return false;
  }
  return false;
}

Maybe<void> InsertNcclLogicalOpPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  auto OpGraphForEachInDataAndCtrlNode = [&](OpNode* node,
                                             const std::function<void(OpNode*)>& Handler) {
    op_graph.ForEachDataAndCtrlInNode(node, Handler);
  };
  auto OpGraphForEachOutDataAndCtrlNode = [&](OpNode* node,
                                              const std::function<void(OpNode*)>& Handler) {
    op_graph.ForEachDataAndCtrlOutNode(node, Handler);
  };

  std::vector<const OpNode*> ordered_op_nodes;
  op_graph.TopoForEachNode(op_graph.DataOrCtrlSourceNodes(), OpGraphForEachInDataAndCtrlNode,
                           OpGraphForEachOutDataAndCtrlNode,
                           [&](const OpNode* node) { ordered_op_nodes.push_back(node); });

  HashSet<const OpNode*> subgraph;
  FindMaxConnectedSubgraphForGpuExecOrder(&subgraph, op_graph, ordered_op_nodes);
  if (subgraph.size() <= 1) { return Maybe<void>::Ok(); }

  std::vector<const OpNode*> subgraph_order;
  HashMap<const OpNode*, int64_t> node2order;
  for (const OpNode* this_node : ordered_op_nodes) {
    if (subgraph.find(this_node) != subgraph.end()) {
      subgraph_order.push_back(this_node);
      node2order.emplace(this_node, subgraph_order.size() - 1);
    }
  }
  CHECK_EQ(subgraph.size(), subgraph_order.size());

  // LOG
  /*
  for (int32_t i = 0; i < subgraph_order.size(); ++i) {
    const OpNode* node = subgraph_order.at(i);
    std::cout << "cclog: i = " << i << ", op_name =  " << node->op().op_name() << std::endl;
  }
  */

  HashSet<std::string> mut_op_names;
  const OpNode* first_node = subgraph_order.at(0);
  HashMap<std::string, OperatorConf> subgraph_op_name2conf;
  subgraph_op_name2conf.emplace(first_node->op().op_name(), first_node->op().op_conf());
  auto IsReachable = op_graph.MakePredicatorIsOpNameDataOrCtrlReachable();
  for (int32_t i = 1; i < subgraph_order.size(); ++i) {
    const OpNode* this_node = subgraph_order.at(i);
    const OpNode* pre_node = subgraph_order.at(i - 1);
    const std::string& this_op_name = this_node->op().op_name();
    const std::string& pre_op_name = pre_node->op().op_name();
    CHECK(subgraph_op_name2conf.emplace(this_op_name, this_node->op().op_conf()).second);
    // build control edge if need.
    if (!IsReachable(pre_op_name, this_op_name)) {
      subgraph_op_name2conf.at(this_op_name).add_ctrl_in_op_name(pre_op_name);
      mut_op_names.insert(this_op_name);

      /*
      std::cout << "cclog: add ctrl edge from  " << pre_op_name << "  to  " << this_op_name
                << std::endl;
      */
    }
  }

  std::vector<OperatorConf> nccl_op_confs;
  for (const OpNode* src_node : subgraph_order) {
    for (const OpEdge* op_edge : src_node->out_edges()) {
      const OpNode* dst_node = op_edge->dst_node();
      const std::string& dst_op_name = dst_node->op().op_name();
      CHECK(src_node != dst_node);
      if (subgraph_op_name2conf.find(dst_op_name) == subgraph_op_name2conf.end()) {
        // NOTE(chengcheng): child node is not in this subgraph.
        continue;
      }
      for (const LogicalBlobId& lbi : op_edge->lbis()) {
        OperatorConf nccl_op;
        if (!TryGetNcclLogicalOpConf(&nccl_op, src_node, dst_node, lbi)) { continue; }
        mut_op_names.insert(dst_op_name);
        // insert nccl op
        user_op::UserOpConfWrapper nccl_op_wrapper(nccl_op);
        for (const std::string& ibn : op_edge->lbi2ibns().at(lbi)) {
          std::string old_lbn = ReplaceInputLbnInOpCustomizedConf(
              &subgraph_op_name2conf.at(dst_op_name), ibn, nccl_op_wrapper.output("out", 0));

          std::cout << "cclog: replace dst_op_name = " << dst_op_name << " input blob name: " << ibn
                    << " from " << old_lbn << "  to  " << nccl_op_wrapper.output("out", 0)
                    << std::endl;
        }

        if (nccl_op_confs.size() >= 1) {
          // NOTE(chengcheng): MUST add ctrl edge between nccl ops for 1 src node insert multi-nccl
          const std::string& pre_nccl_op_name = nccl_op_confs.at(nccl_op_confs.size() - 1).name();
          nccl_op.add_ctrl_in_op_name(pre_nccl_op_name);

          /*
          std::cout << "cclog: add ctrl edge from  " << pre_nccl_op_name << "  to  "
                    << nccl_op.name() << std::endl;
          */
        }

        // NOTE(chengcheng): src_node MUST not the last node in subgraph, find the next op
        int64_t src_order = node2order.at(src_node);
        CHECK(src_order + 1 < subgraph_order.size());
        const std::string& next_op_name = subgraph_order.at(src_order + 1)->op().op_name();
        if (dst_op_name != next_op_name) {
          // NOTE(chengcheng): MUST add ctrl edge for strict exec order
          subgraph_op_name2conf.at(next_op_name).add_ctrl_in_op_name(nccl_op.name());
          mut_op_names.insert(next_op_name);

          /*
          std::cout << "cclog: add ctrl edge from  " << nccl_op.name() << "  to  " << next_op_name
                    << std::endl;
                    */
        }

        nccl_op_confs.push_back(nccl_op);
      }
    }
  }

  std::vector<OperatorConf> mut_op_confs;
  for (const std::string& mut_op_name : mut_op_names) {
    // std::cout << "cclog: mut op name = " << mut_op_name << std::endl;
    mut_op_confs.push_back(subgraph_op_name2conf.at(mut_op_name));
  }
  job_builder->MutOpsOnlyOnce(mut_op_confs);
  job_builder->AddOps(first_node->parallel_desc().parallel_conf(), nccl_op_confs);

  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("InsertNcclLogicalOpPass", InsertNcclLogicalOpPass);

}  // namespace oneflow

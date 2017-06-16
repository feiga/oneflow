#ifndef ONEFLOW_CORE_OPERATOR_CPU_CLEAR_OP_H_
#define ONEFLOW_CORE_OPERATOR_CPU_CLEAR_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/register_desc.h"

namespace oneflow {

class CpuClearOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuClearOp);
  CpuClearOp() = default;
  ~CpuClearOp() = default;

  void InitFromOpConf(const OperatorConf& op_conf) override;

  const PbMessage& GetSpecialConf() const override;
  
 private:

};

} // namespace oneflow

#endif // ONEFLOW_CORE_OPERATOR_CPU_CLEAR_OP_H_

include(ExternalProject)
set(CARES_TAR_URL https://github.com/c-ares/c-ares/releases/download/cares-1_15_0/c-ares-1.15.0.tar.gz)
use_mirror(VARIABLE CARES_TAR_URL URL ${CARES_TAR_URL})
set(CARES_URL_HASH d2391da274653f7643270623e822dff7)
set(CARES_INSTALL ${THIRD_PARTY_DIR}/cares)
SET(CARES_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/cares/src/cares)

set(CARES_INCLUDE_DIR ${THIRD_PARTY_DIR}/cares/include)
set(CARES_LIBRARY_DIR ${THIRD_PARTY_DIR}/cares/lib)

if(WIN32)
  set(CARES_BUILD_LIBRARY_DIR ${CARES_INSTALL}/lib)
  set(CARES_LIBRARY_NAMES cares.lib)
elseif(APPLE AND ("${CMAKE_GENERATOR}" STREQUAL "Xcode"))
  set(CARES_BUILD_LIBRARY_DIR ${CARES_INSTALL}/lib)
  set(CARES_LIBRARY_NAMES libcares.a)
else()
  set(CARES_BUILD_LIBRARY_DIR ${CARES_INSTALL}/lib)
  set(CARES_LIBRARY_NAMES libcares.a)
endif()

foreach(LIBRARY_NAME ${CARES_LIBRARY_NAMES})
  list(APPEND CARES_STATIC_LIBRARIES ${CARES_LIBRARY_DIR}/${LIBRARY_NAME})
  list(APPEND CARES_BUILD_STATIC_LIBRARIES ${CARES_BUILD_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

if(THIRD_PARTY)
ExternalProject_Add(cares
  PREFIX cares 
  URL ${CARES_TAR_URL}
  URL_HASH MD5=${CARES_URL_HASH}
  UPDATE_COMMAND ""
  CMAKE_CACHE_ARGS
      -DCARES_STATIC:BOOL=ON 
      -DCARES_SHARED:BOOL=OFF
      -DCARES_STATIC_PIC:BOOL=ON
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX:STRING=${CARES_INSTALL}
      -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
)

endif()

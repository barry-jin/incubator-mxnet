/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
*  Copyright (c) 2016 by Contributors
* \file op_rt_map.h
* \brief definition of runtime operator map
* \author Zhenghui Jin
*/

#ifndef MXNET_CPP_OP_RT_MAP_H_
#define MXNET_CPP_OP_RT_MAP_H_

#include <map>
#include <string>
#include "mxnet-cpp/base.h"
#include "dmlc/logging.h"

namespace mxnet {
namespace cpp {

/*!
 * \brief OpMap instance holds a map of all the symbol creators so we can
 *  get symbol creators by name.
 *  This is used internally by Symbol and Operator.
 */
class OpRTMap {
 public:
  /*!
  * \brief Create an Operator runtime API map
  */
  inline OpRTMap() {
    int out_size;
    const char** op_names;
    int err = MXNetFuncListGlobalNames(&out_size, &op_names);
    if (err != 0) {
      LOG(FATAL) << MXGetLastError();
    }
    for (nn_uint i = 0; i < out_size; i++) {
      MXNetFunctionHandle handle;
      int err = MXNetFuncGetGlobal(op_names[i], &handle);
      if (err != 0) {
        LOG(FATAL) << MXGetLastError();
      }
      std::string op_name = op_names[i];
      size_t pos = op_name.find_last_of(".");
      if (pos != std::string::npos) {
        op_handles_[op_name.substr(pos+1)] = handle;
      } else {
        op_handles_[op_names[i]] = handle;
      }
    }
  }
  /*!
  * \brief destructor
  */
  virtual ~OpRTMap() {
    for (auto const& entry : op_handles_) {
      int err = MXNetFuncFree(entry.second);
      if (err != 0) {
        LOG(FATAL) << MXGetLastError();
      }
    }
  };

  /*!
  * \brief Get an op handle with operator name.
  *
  * \param name name of the op
  * \return handle to the op
  */
  inline MXNetFunctionHandle GetOpHandle(const std::string &name) {
    return op_handles_[name];
  }

 private:
  std::map<std::string, MXNetFunctionHandle> op_handles_;
};

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_OP_RT_MAP_H_

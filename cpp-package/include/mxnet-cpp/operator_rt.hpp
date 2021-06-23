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
* \file operator_rt.hpp
* \brief implementation of operator runtime API
* \author Zhenghui Jin
*/

#ifndef MXNET_CPP_OPERATOR_RT_HPP_
#define MXNET_CPP_OPERATOR_RT_HPP_

#include <algorithm>
#include <string>
#include <vector>
#include <iterator>
#include "mxnet-cpp/base.h"
#include "mxnet-cpp/op_rt_map.h"
#include "mxnet-cpp/operator_rt.h"

namespace mxnet {
namespace cpp {

inline OpRTMap*& OperatorRT::op_map() {
  static OpRTMap *op_map_ = new OpRTMap();
  return op_map_;
}

inline OperatorRT::OperatorRT(const std::string &operator_name) {
  _handle = op_map()->GetOpHandle(operator_name);
  _value = new MXNetValue();
}

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_OPERATOR_RT_HPP_

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
* \file adt.hpp
* \brief implementation of adt
* \author Zhenghui Jin
*/

#ifndef MXNET_CPP_ADT_HPP_
#define MXNET_CPP_ADT_HPP_

#include <algorithm>
#include <string>
#include <vector>
#include <iterator>
#include "mxnet-cpp/base.h"
#include "mxnet-cpp/container/adt.h"

namespace mxnet {
namespace cpp {

inline OpRTMap*& ADT::op_map() {
  static OpRTMap *op_map_ = new OpRTMap();
  return op_map_;
}

inline ADT::ADT(void* object_handle) {
  _adt = op_map()->GetOpHandle("_ADT");
  _value = new MXNetValue();
  _value->v_handle = object_handle;
  type_code = kObjectHandle;
  MXNetFunctionHandle _get_size = op_map()->GetOpHandle("_GetADTSize");
  MXNetValue* temp = new MXNetValue();
  int temp_type = kObjectHandle;
  CHECK_EQ(MXNetFuncCall(_get_size, _value, &type_code, 1, temp, &temp_type), 0);
  adt_size = temp->v_int64;
}

inline ADT::ADT(Shape s) {
  _adt = op_map()->GetOpHandle("_ADT");
  _value = new MXNetValue();
  adt_size = s.ndim();
  std::vector<MXNetValue> input_fields;
  std::vector<int> input_types;
  for (int i = 0; i < s.ndim(); i++) {
    MXNetValue cur_field = IntegerObj(s[i]);
    input_fields.push_back(cur_field);
    input_types.push_back(kObjectHandle);
  }
  CHECK_EQ(MXNetFuncCall(_adt, input_fields.data(), input_types.data(), input_types.size(), _value, &type_code), 0);
}

MXNetValue ADT::IntegerObj(int arg) {
  MXNetFunctionHandle _integer = op_map()->GetOpHandle("_Integer");
  MXNetValue* temp = new MXNetValue();
  MXNetValue* ret = new MXNetValue();
  temp->v_int64 = arg;
  int type_codes_int = kDLInt;
  int ret_type_code_int;
  CHECK_EQ(MXNetFuncCall(_integer, temp, &type_codes_int, 1, ret, &ret_type_code_int), 0);
  return *ret;
}

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_ADT_HPP_

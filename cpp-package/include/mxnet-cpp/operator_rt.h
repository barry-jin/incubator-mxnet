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
* \file operator_rt.h
* \brief Operator Runtime API
* \author Zhenghui Jin
*/

#ifndef MXNET_CPP_OPERATOR_RT_H_
#define MXNET_CPP_OPERATOR_RT_H_

#include <map>
#include <string>
#include <vector>
#include "dmlc/logging.h"
#include "mxnet-cpp/base.h"
#include "mxnet-cpp/op_rt_map.h"
#include "mxnet-cpp/ndarray.h"
#include "mxnet-cpp/shape.h"
#include "mxnet-cpp/packed_func.h"
#include "mxnet-cpp/container/adt.h"

/*!
 * \brief Macro for Operator definition in op namespace
 * \param name name of the operator
 */
#define MXNET_DEFINE_OP(name) \
namespace mxnet { \
namespace cpp { \
namespace op { \
  OperatorRT name = OperatorRT(#name); \
} \
} \
}


namespace mxnet {
namespace cpp {
/*!
* \brief Operator interface
*/
class OperatorRT {
 public:
  /*!
   * \brief Operator constructor
   * \param operator_name type of the operator
   */
  explicit OperatorRT(const std::string &operator_name);
  OperatorRT &operator=(const OperatorRT &rhs);
  /*!
   * \brief Create Operator Args
   * \param arg input arg
   */
  template <typename T>
  inline void create_args(T arg) {
    PushArgs(arg);
  }
  /*!
   * \brief Create Operator Args
   * \param arg input arg
   */
  template <typename T, typename... Args>
  inline void create_args(T arg, Args... args) {
    PushArgs(arg);
    create_args(args...);
  }
  /*!
   * \brief Push args as MXNetValue
   * \param arg input arg with type int
   */
  void PushArgs(int arg) {
    MXNetValue* temp = new MXNetValue();
    temp->v_int64 = arg;
    arg_values.push_back(*temp);
    type_codes.push_back(kDLInt);
  }
  /*!
   * \brief Push args as MXNetValue
   * \param arg input arg with type float
   */
  void PushArgs(float arg) {
    MXNetValue* temp = new MXNetValue();
    temp->v_float64 = arg;
    arg_values.push_back(*temp);
    type_codes.push_back(kDLFloat);
  }
  /*!
   * \brief Push args as MXNetValue
   * \param arg input arg with type double
   */
  void PushArgs(double arg) {
    MXNetValue* temp = new MXNetValue();
    temp->v_float64 = arg;
    arg_values.push_back(*temp);
    type_codes.push_back(kDLFloat);
  }
  /*!
   * \brief Push args as MXNetValue
   * \param arg input arg with type NDArray
   */
  void PushArgs(NDArray arg) {
    MXNetValue* temp = new MXNetValue();
    temp->v_handle = arg.GetHandle();
    arg_values.push_back(*temp);
    type_codes.push_back(kNDArrayHandle);
  }
  /*!
   * \brief Push args as MXNetValue
   * \param arg input arg with type char*
   */
  void PushArgs(const char* arg) {
    MXNetValue* temp = new MXNetValue();
    temp->v_str = arg;
    arg_values.push_back(*temp);
    type_codes.push_back(kStr);
  }
  /*!
   * \brief Push args as MXNetValue
   * \param arg input arg with type void*
   */
  void PushArgs(void* arg) {
    MXNetValue* temp = new MXNetValue();
    temp->v_handle = arg;
    arg_values.push_back(*temp);
    type_codes.push_back(kHandle);
  }
  /*!
   * \brief Push args as MXNetValue
   * \param arg input arg with type nullptr_t
   */
  void PushArgs(nullptr_t arg) {
    MXNetValue* temp = new MXNetValue();
    temp->v_handle = nullptr;
    arg_values.push_back(*temp);
    type_codes.push_back(kNull);
  }
  /*!
   * \brief Push args as MXNetValue
   * \param arg input arg with type Shape
   */
  void PushArgs(Shape arg) {
    ADT* adt = new ADT(arg);
    MXNetValue* temp = new MXNetValue();
    temp->v_handle = adt->GetPointer();
    arg_values.push_back(*temp);
    type_codes.push_back(kObjectHandle);
  }
  /*!
   * \brief Make Operator Call
   * \param args input args
   */
  template<typename... Args>
  MXNetValue_ operator()(Args... args) {
    create_args(args...);
    int err = MXNetFuncCall(_handle, arg_values.data(), type_codes.data(), type_codes.size(), _value, &ret_type_code);
    if (err != 0) {
      LOG(FATAL) << MXGetLastError();
    }
    MXNetValue_ ret;
    if (ret_type_code == kPyArg) {
      MXNetValue* temp = new MXNetValue(arg_values[_value->v_int64]);
      ret = MXNetValue_(*temp, type_codes[_value->v_int64]);
    } else {
      ret = MXNetValue_(*_value, ret_type_code);
    }
    arg_values.clear();
    type_codes.clear();
    return ret;
  }

 private:
  static OpRTMap*& op_map();
  MXNetFunctionHandle _handle;
  std::vector<MXNetValue> arg_values;
  std::vector<int> type_codes;
  MXNetValue* _value;
  int ret_type_code;
};
}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_OPERATOR_RT_H_

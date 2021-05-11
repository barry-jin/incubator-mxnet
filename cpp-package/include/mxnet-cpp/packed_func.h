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
* \file packed_func.h
* \brief definition of packed function
* \author Zhenghui Jin
*/

#ifndef MXNET_CPP_UTILS_H_
#define MXNET_CPP_UTILS_H_

#include <dmlc/strtonum.h>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include "mxnet-cpp/base.h"
#include "dmlc/logging.h"
#include "mxnet-cpp/ndarray.h"
#include "mxnet-cpp/op_rt_map.h"
#include "mxnet-cpp/container/adt.h"

namespace mxnet {
namespace cpp {

// implementation details
inline const char* TypeCode2Str(int type_code) {
  switch (type_code) {
    case kDLInt: return "int";
    case kDLUInt: return "uint";
    case kDLFloat: return "float";
    case kStr: return "str";
    case kBytes: return "bytes";
    case kHandle: return "handle";
    case kNull: return "NULL";
    case kObjectHandle: return "ObjectCell";
    case kNDArrayHandle: return "NDArray";
    default: LOG(FATAL) << "unknown type_code="
                        << static_cast<int>(type_code); return "";
  }
}

// macro to check type code.
#define MXNET_CHECK_TYPE_CODE(CODE, T)                           \
  CHECK_EQ(CODE, T) << " expected "                            \
  << TypeCode2Str(T) << " but get " << TypeCode2Str(CODE)      \

/*!
 * \brief Internal base class to
 *  handle conversion from MXNetValue.
 */
class MXNetValue_ {
 public:
  operator double() const {
    // Allow automatic conversion from int to float
    // This avoids errors when user pass in int from
    // the frontend while the API expects a float.
    if (type_code_ == kDLInt) {
      return static_cast<double>(value_.v_int64);
    }
    MXNET_CHECK_TYPE_CODE(type_code_, kDLFloat);
    return value_.v_float64;
  }
  operator int64_t() const {
    MXNET_CHECK_TYPE_CODE(type_code_, kDLInt);
    return value_.v_int64;
  }
  operator uint64_t() const {
    MXNET_CHECK_TYPE_CODE(type_code_, kDLInt);
    return value_.v_int64;
  }
  operator int() const {
    MXNET_CHECK_TYPE_CODE(type_code_, kDLInt);
    CHECK_LE(value_.v_int64,
             std::numeric_limits<int>::max());
    return static_cast<int>(value_.v_int64);
  }
  operator bool() const {
    MXNET_CHECK_TYPE_CODE(type_code_, kDLInt);
    return value_.v_int64 != 0;
  }
  operator void*() const {
    if (type_code_ == kNull) return nullptr;
    MXNET_CHECK_TYPE_CODE(type_code_, kHandle);
    return value_.v_handle;
  }
  operator NDArray() const {
    MXNET_CHECK_TYPE_CODE(type_code_, kNDArrayHandle);
    NDArray* ret = new NDArray(value_.v_handle);
    return *ret;
  }
  operator std::vector<NDArray>() const {
    MXNET_CHECK_TYPE_CODE(type_code_, kObjectHandle);
    ADT* adt = new ADT(value_.v_handle);
    index_t adt_size = adt->GetSize();
    std::vector<NDArray> ret;
    for (int i = 0; i < adt_size; i++) {
      MXNetValue cur_val = (*adt)[i];
      NDArray* cur_arr = new NDArray(cur_val.v_handle);
      ret.push_back(*cur_arr);
    }
    return ret;
  }
  template<typename TObjectRef,
           typename = typename std::enable_if<
             std::is_class<TObjectRef>::value>::type>
  inline bool IsObjectRef() const;
  template <typename TObjectRef>
  inline TObjectRef AsObjectRef() const;
  int type_code() const {
    return type_code_;
  }

  /*!
   * \brief return handle as specific pointer type.
   * \tparam T the data type.
   * \return The pointer type.
   */
  template<typename T>
  T* ptr() const {
    return static_cast<T*>(value_.v_handle);
  }

  MXNetValue_() : type_code_(kNull) {}
  MXNetValue_(MXNetValue value, int type_code)
      : value_(value), type_code_(type_code) {}
 protected:
  /*! \brief The value */
  MXNetValue value_;
  /*! \brief the type code */
  int type_code_;
};

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_UTILS_H_

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
* \file adt.h
* \brief definition of adt
* \author Zhenghui Jin
*/

#ifndef MXNET_CPP_ADT_H_
#define MXNET_CPP_ADT_H_

#include <map>
#include <string>
#include <vector>
#include "mxnet-cpp/base.h"
#include "mxnet-cpp/shape.h"
#include "mxnet-cpp/op_rt_map.h"

namespace mxnet {
namespace cpp {
/*!
* \brief Algebatic data type(ADT) object.
*/
class ADT {
 public:
  /*!
  * \brief ADT constructor
  * \param object_handle ADT object handle
  */
  explicit ADT(void* object_handle);
  /*!
  * \brief constructor from shape
  * \param s input shape
  */
  explicit ADT(Shape s);
  /*!
  * \brief destructor
  */
  virtual ~ADT() {
    CHECK_EQ(MXNetObjectFree(_value->v_handle), 0);
  };
  /*!
  * \brief create integer object
  * \param arg input integer type
  */
  MXNetValue IntegerObj(int arg);
  /*!
  * \brief get adt object pointer
  */
  void* GetPointer() {
    return _value->v_handle;
  }
  /*!
  * \brief get adt object pointer
  */
  index_t GetSize() {
    return adt_size;
  }
  /*!
  * \brief get corresponding index
  * \param i dimension index
  * \return the corresponding NDArray
  */
  inline MXNetValue &operator[](index_t i) {
    std::vector<MXNetValue> temp_vec;
    std::vector<int> temp_types;
    MXNetFunctionHandle _get_fields = op_map()->GetOpHandle("_GetADTFields");
    MXNetValue* temp = new MXNetValue();
    MXNetValue* idx = new MXNetValue();
    idx->v_int64 = i;
    int temp_type = kObjectHandle;
    temp_vec.push_back(*_value);
    temp_types.push_back(type_code);
    temp_vec.push_back(*idx);
    temp_types.push_back(kDLInt);
    CHECK_EQ(MXNetFuncCall(_get_fields, temp_vec.data(), temp_types.data(), 2, temp, &temp_type), 0);
    return *temp;
  }

 private:
  static OpRTMap*& op_map();
  MXNetFunctionHandle _adt;
  MXNetValue* _value;
  int type_code;
  int adt_size;
};
}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_ADT_H_

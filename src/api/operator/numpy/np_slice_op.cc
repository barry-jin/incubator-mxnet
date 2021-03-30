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
 * \file np_slice_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_clice_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/tensor/slice-inl.h"
#include "../../../operator/tensor/matrix_op-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.slice")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npi_slice");
  op::SliceParam param;
  // begin tuple
  param.begin = Tuple<dmlc::optional<index_t>>(args[1].operator ObjectRef());
  // end tuple
  param.end = Tuple<dmlc::optional<index_t>>(args[2].operator ObjectRef());
  // step tuple
  if (args[3].type_code() == kNull) {
    param.step = Tuple<dmlc::optional<index_t>>();
  } else {
    param.step = Tuple<dmlc::optional<index_t>>(args[3].operator ObjectRef());
  }
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::SliceParam>(&attrs);
  // inputs
  NDArray* inputs[] = {args[0].operator NDArray*()};
  int num_inputs = 1;
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

}  // namespace mxnet

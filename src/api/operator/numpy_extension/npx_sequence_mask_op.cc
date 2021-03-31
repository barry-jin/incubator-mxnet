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
 * \file npx_sequence_mask_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy_extension/npx_sequence_mask_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/sequence_mask-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npx.sequence_mask")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  static const nnvm::Op* op = Op::Get("_npx_sequence_mask");
  op::SequenceMaskParam param;
  int args_size = args.size();
  // inputs
  int num_inputs = args_size - 3;
  std::vector<NDArray*> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(args[i].operator mxnet::NDArray*());
  }

  // parse use_sequence_length
  if (args[args_size - 3].type_code() == kNull) {
    param.use_sequence_length = false;
  } else {
    param.use_sequence_length = args[args_size - 3].operator bool();
  }

  // parse value
  if (args[args_size - 2].type_code() == kNull) {
    param.value = 0.0;
  } else {
    param.value = args[args_size - 2].operator double();
  }

  // parse axis
  if (args[args_size - 1].type_code() == kNull) {
    param.axis = 0;
  } else {
    param.axis = args[args_size - 1].operator int();
  }

  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::SequenceMaskParam>(&attrs);

  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs.data(), &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

}  // namespace mxnet

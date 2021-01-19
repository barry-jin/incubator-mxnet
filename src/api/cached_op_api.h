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
 *  Copyright (c) 2019 by Contributors
 * \file cached_op_api.h
 * \brief Function definition of cachedOp apis
 */

#ifndef MXNET_API_CACHED_OP_API_H_
#define MXNET_API_CACHED_OP_API_H_

#include "../imperative/cached_op.h"
#include "../imperative/cached_op_threadsafe.h"

namespace mxnet {

void InvokeCachedOpImpl(CachedOpHandle handle,
                        int num_inputs,
                        NDArrayHandle **inputs,
                        int default_dev_type,
                        int default_dev_id,
                        int *num_outputs,
                        NDArrayHandle ***outputs,
                        const int **out_stypes) {
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  CachedOpPtr op_shared = *static_cast<CachedOpPtr*>(handle);
  // CachedOp* points to CachedOpThreadSafe object if CreateCachedOpEX
  // was called with thread_safe=true
  CachedOp* op = dynamic_cast<CachedOp*>(op_shared.get());
  std::vector<NDArray*> ndinputs;
  ndinputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    ndinputs.push_back(reinterpret_cast<NDArray*>(inputs[i]));
  }

  std::vector<NDArray*> ndoutputs;
  ndoutputs.reserve(op->num_outputs());
  if (*outputs == nullptr) {
    *num_outputs = op->num_outputs();
    for (int i = 0; i < *num_outputs; ++i) ndoutputs.push_back(new NDArray());
  } else {
    CHECK_EQ(*num_outputs, op->num_outputs())
        << "CachedOp expects " << op->num_outputs() << " outputs, but "
        << *num_outputs << " was given.";
    for (int i = 0; i < *num_outputs; ++i) {
      ndoutputs.push_back(reinterpret_cast<NDArray*>((*outputs)[i]));
    }
  }
  // construct default context
  Context ctx = Context::Create(static_cast<Context::DeviceType>(default_dev_type),
                                default_dev_id);
  op->Forward(op_shared, ndinputs, ndoutputs, ctx);

  if (*outputs == nullptr) {
    ret->ret_handles.clear();
    ret->ret_handles.reserve(*num_outputs);
    for (int i = 0; i < *num_outputs; ++i) {
      ret->ret_handles.push_back(ndoutputs[i]);
    }
    *outputs = reinterpret_cast<NDArrayHandle**>(dmlc::BeginPtr(ret->ret_handles));
  }

  NDArray** out_array = reinterpret_cast<NDArray**>(*outputs);
  ret->out_types.clear();
  ret->out_types.reserve(*num_outputs);
  for (int i = 0; i < *num_outputs; ++i) {
    ret->out_types.emplace_back(out_array[i]->storage_type());
  }
  *out_stypes = dmlc::BeginPtr(ret->out_types);
}

}  // namespace mxnet

#endif  // MXNET_API_CACHED_OP_API_H_

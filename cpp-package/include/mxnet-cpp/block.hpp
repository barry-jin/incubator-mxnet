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
* \file block.hpp
* \brief implementation of block
* \author Zhenghui Jin
*/

#ifndef MXNET_CPP_GLUON_BLOCK_HPP_
#define MXNET_CPP_GLUON_BLOCK_HPP_

#include <algorithm>
#include <string>
#include <vector>
#include <iterator>
#include "mxnet-cpp/base.h"
#include "mxnet-cpp/block.h"

namespace mxnet {
namespace cpp {
namespace gluon {

template <typename BlockType>
BlockType Block::register_block(std::string name, BlockType block) {
  _children_vec.emplace_back(
      std::make_pair<std::string, std::shared_ptr<BlockType>>(std::move(name), std::make_shared<BlockType>(block))
  );
  return *(std::dynamic_pointer_cast<BlockType>(_children_vec.back().second).get());
}

NDArray Block::register_parameter(std::string name,
                                  Shape shape,
                                  OpReqType grad_req,
                                  Context ctx) {
  _param_vec.emplace_back(
      std::make_tuple(std::move(name), NDArray(shape, ctx), NDArray(shape, ctx), grad_req)
  );
  return std::get<1>(_param_vec.back());
}

Block::paramType Block::collect_parameters(std::string prefix) {
  if (!prefix.empty()) {
    prefix += '.';
  }
  Block::paramType ret;
  for (auto& entry : _param_vec) {
    ret.emplace_back(
        std::make_tuple(prefix + std::get<0>(entry),
                       static_cast<NDArray>(std::get<1>(entry)),
                       static_cast<NDArray>(std::get<2>(entry)),
                       std::get<3>(entry))
    );
  }
  for (auto& entry : _children_vec) {
    Block::paramType child =
        entry.second->collect_parameters(prefix + entry.first);
    ret.insert(ret.end(), child.begin(), child.end());
  }
  return ret;
}

template <typename InitType>
void Block::initialize(InitType init, Context ctx) {
  this->device_type = ctx.GetDeviceType();
  this->device_id = ctx.GetDeviceId();
  Block::paramType params =
      Block::collect_parameters("");
  for (auto& arg : params) {
    init(std::get<0>(arg), &std::get<1>(arg));
  }
  std::vector<NDArrayHandle> arg_handles;
  std::vector<NDArrayHandle> grad_handles;
  std::vector<mx_uint> grad_reqs_uint;
  for (int i = 0; i < params.size(); i++) {
    arg_handles.emplace_back(std::get<1>(params[i]).GetHandle());
    grad_handles.emplace_back(std::get<2>(params[i]).GetHandle());
    grad_reqs_uint.emplace_back(std::get<3>(params[i]));
  }
  CHECK_EQ(MXAutogradMarkVariables(arg_handles.size(), arg_handles.data(),
                                    grad_reqs_uint.data(), grad_handles.data()),0);
}

void Block::initialize(Context ctx) {
  this->device_type = ctx.GetDeviceType();
  this->device_id = ctx.GetDeviceId();
}

}  // namespace gluon
}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_GLUON_BLOCK_HPP_

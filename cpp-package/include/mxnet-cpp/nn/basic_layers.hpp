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
* \file basic_layers.hpp
* \brief definition of basic_layers
* \author Zhenghui Jin
*/

#ifndef MXNET_CPP_NN_BASIC_LAYERS_HPP_
#define MXNET_CPP_NN_BASIC_LAYERS_HPP_

#include <dmlc/strtonum.h>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include "mxnet-cpp/base.h"
#include "dmlc/logging.h"
#include "mxnet-cpp/nn/basic_layers.h"

MXNET_DEFINE_OP(fully_connected)
MXNET_DEFINE_OP(activation)
MXNET_DEFINE_OP(copy)
MXNET_DEFINE_OP(dropout)

namespace mxnet {
namespace cpp {
namespace gluon {
namespace nn {

inline Dense::Dense(int units, int in_units, std::string act_type) {
  this->units = units;
  this->weight = register_parameter("weight", Shape(units, in_units));
  this->bias = register_parameter("bias", Shape(units));
  this->act_type = act_type;
}

inline NDArray Dense::hybrid_forward(NDArray arr) {
  NDArray ret = op::fully_connected(arr, this->weight, this->bias, this->units, false, true);
  if (this->act_type != "None") {
    Activation act(this->act_type);
    return act(ret);
  }
  return ret;
}

inline Activation::Activation(std::string act_type) {
  this->_act_type = act_type;
}

inline NDArray Activation::hybrid_forward(NDArray arr) {
  return op::activation(arr, (this->_act_type).c_str());
}

inline Dropout::Dropout(double rate) {
  this->_rate = rate;
}

inline NDArray Dropout::hybrid_forward(NDArray arr) {
  if (this->_rate <= 0) {
    return op::copy(arr);
  } else {
    return op::dropout(arr, this->_rate, "training", nullptr, false);
  }
}

}  // namespace nn
}  // namespace gluon
}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_NN_BASIC_LAYERS_HPP_
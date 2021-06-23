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
* \file basic_layers.h
* \brief definition of basic_layers
* \author Zhenghui Jin
*/

#ifndef MXNET_CPP_NN_BASIC_LAYERS_H_
#define MXNET_CPP_NN_BASIC_LAYERS_H_

#include <dmlc/strtonum.h>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include "mxnet-cpp/base.h"
#include "dmlc/logging.h"
#include "mxnet-cpp/ndarray.h"
#include "mxnet-cpp/block.h"
#include "mxnet-cpp/operator_rt.h"


namespace mxnet {
namespace cpp {
namespace gluon {
namespace nn {

/*!
* \brief Dense Layer
*/
class Dense : public Block {
 public:
  Dense() {}
  explicit Dense(int units, int in_units = 0, std::string act_type = "None");
  NDArray hybrid_forward(NDArray arr);
 private:
  NDArray weight;
  NDArray bias;
  int units;
  std::string act_type;
};

/*!
* \brief Activation
*/
class Activation : public Block {
 public:
  Activation() {}
  explicit Activation(std::string act_type = "relu");
  NDArray hybrid_forward(NDArray arr);
 private:
  std::string _act_type;
};

/*!
* \brief Dropout
*/
class Dropout : public Block {
 public:
  explicit Dropout(double rate = 0.5);
  NDArray hybrid_forward(NDArray arr);
 private:
  double _rate;
};


}  // namespace nn
}  // namespace gluon
}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_NN_BASIC_LAYERS_H_
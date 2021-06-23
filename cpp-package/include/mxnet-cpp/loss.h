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
* \file loss.h
* \brief definition of loss functions
* \author Zhenghui Jin
*/

#ifndef MXNET_CPP_LOSS_H_
#define MXNET_CPP_LOSS_H_

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

/*!
* \brief Dense Layer
*/
class SoftmaxCrossEntropyLoss : public Block {
 public:
  SoftmaxCrossEntropyLoss() {
    this->initialize();
    this->need_cache = false;
  }
  NDArray hybrid_forward(NDArray pred, NDArray label);
};


}  // namespace gluon
}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_LOSS_H_
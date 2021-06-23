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

#ifndef MXNET_CPP_LOSS_HPP_
#define MXNET_CPP_LOSS_HPP_

#include <dmlc/strtonum.h>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include "mxnet-cpp/base.h"
#include "dmlc/logging.h"
#include "mxnet-cpp/loss.h"

MXNET_DEFINE_OP(log_softmax)
MXNET_DEFINE_OP(pick)
MXNET_DEFINE_OP(negative)

namespace mxnet {
namespace cpp {
namespace gluon {

inline NDArray SoftmaxCrossEntropyLoss::hybrid_forward(NDArray pred, NDArray label) {
  NDArray pred_new = op::log_softmax(pred, -1, nullptr, false, "float32");
  NDArray loss = op::pick(pred_new, label, -1, "clip", true);
  NDArray neg = op::negative(loss, nullptr);
  NDArray ret = op::mean(neg, 0, nullptr, false, nullptr);
  return ret;
}


}  // namespace gluon
}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_LOSS_HPP_
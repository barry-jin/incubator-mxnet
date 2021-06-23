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
* \file trainer.h
* \brief definition of trainer
* \author Zhenghui Jin
*/

#ifndef MXNET_CPP_TRAINER_H_
#define MXNET_CPP_TRAINER_H_

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
#include "mxnet-cpp/optimizer.h"


namespace mxnet {
namespace cpp {

/*!
* \brief Trainer interface
*/
class Trainer {
 public:
  /*!
   * \brief constructor
   */
  Trainer(std::vector<std::tuple<std::string, NDArray, NDArray, OpReqType>> parameters,
          const std::string& optimizer,
          const std::unordered_map<std::string, double>& optimizer_params) {
    _parameters = parameters;
    _opt = OptimizerRegistry::Find(optimizer);
    for (auto& entry : optimizer_params) {
      _opt->SetParam(entry.first, entry.second);
    }
  }
  /*!
   * \brief destructor
   */
  virtual ~Trainer() = default;
  /*!
   * \brief start autograd recording
   */
  void step() {
    for (size_t i = 0; i < _parameters.size(); ++i) {
      _opt->Update(i, std::get<1>(_parameters[i]), std::get<2>(_parameters[i]));
    }
  }
 private:
  std::vector<std::tuple<std::string, NDArray, NDArray, OpReqType>> _parameters;
  Optimizer* _opt;
};

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_TRAINER_H_
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
* \file autograd.h
* \brief definition of autograd
* \author Zhenghui Jin
*/

#ifndef MXNET_CPP_AUTOGRAD_H_
#define MXNET_CPP_AUTOGRAD_H_

#include <dmlc/strtonum.h>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include "mxnet-cpp/base.h"
#include "dmlc/logging.h"
#include "mxnet-cpp/ndarray.h"


namespace mxnet {
namespace cpp {

/*!
* \brief AutoGrad interface
*/
class AutoGrad {
 public:
  /*!
   * \brief constructor
   */
  AutoGrad(int is_record, int train_mode) {
    this->_enter_is_record = is_record;
    this->_enter_train_mode = train_mode;
    this->_prev_is_record = -1;
    this->_prev_train_mode = -1;
  }
  /*!
   * \brief destructor
   */
  virtual ~AutoGrad() = default;
  /*!
   * \brief start autograd recording
   */
  void start_recording() {
    if (_enter_is_record != -1) {
      CHECK_EQ(MXAutogradSetIsRecording(_enter_is_record, &_prev_is_record), 0);
    }
    if (_enter_train_mode != -1) {
      CHECK_EQ(MXAutogradSetIsTraining(_enter_train_mode, &_prev_train_mode), 0);
    }
  }
  /*!
   * \brief finish autograd recording
   */
  void finish_recording() {
    if (_enter_is_record != -1 && _prev_is_record != _enter_is_record) {
      CHECK_EQ(MXAutogradSetIsRecording(_prev_is_record, &_prev_is_record), 0);
    }
    if (_enter_train_mode != -1 && _prev_train_mode != _enter_train_mode) {
      CHECK_EQ(MXAutogradSetIsTraining(_prev_train_mode, &_prev_train_mode), 0);
    }
  }
  /*!
   * \brief Backward
   * \param name name of the children block
   * \param block the pointer to the child block
   * \return reference to the child block
   */
  void backward(const NDArray& head) {
    std::vector<NDArrayHandle> head_handle;
    head_handle.push_back(head.GetHandle());
    CHECK_EQ(MXAutogradBackwardEx(1, head_handle.data(),
                                  nullptr, 0, nullptr, 0, 0, 1,
                                  nullptr, nullptr), 0);
  }
 private:
  int _enter_is_record;
  int _enter_train_mode;
  int _prev_is_record;
  int _prev_train_mode;
};

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_AUTOGRAD_H_
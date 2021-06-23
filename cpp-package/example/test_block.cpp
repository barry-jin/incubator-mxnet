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
 * Xin Li yakumolx@gmail.com
 * The file is used for testing if the score(accurary) we get
 * is better than the threshold we set using mlp model.
 * By running: build/test_score 0.75
 * 0.75 here means the threshold score
 * It return 0 if we can achieve higher score than threshold, otherwise 1
 */
#include <chrono>
#include <string>
#include "utils.h"
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

MXNET_DEFINE_OP(ones)
MXNET_DEFINE_OP(zeros)


class Model : public gluon::Block {
 public:
  Model() {
    dense0 = register_block("dense0", gluon::nn::Dense(64, 784, "relu"));
    // dropout0 = register_block("dropout0", gluon::nn::Dropout(0.5));
    dense1 = register_block("dense1", gluon::nn::Dense(32, 64, "relu"));
    dense2 = register_block("dense2", gluon::nn::Dense(10, 32));
  }

  NDArray forward(NDArray x) {
    x = dense0(x);
    // x = dropout0(x);
    x = dense1(x);
    x = dense2(x);
    return x;
  }
  gluon::nn::Dense dense0, dense1, dense2;
  // gluon::nn::Dropout dropout0;
};

int main(int argc, char** argv) {
  Model model;
  model.initialize<Uniform>(Uniform(0.07));
  NDArray data = op::ones(Shape(12, 784), nullptr, nullptr);
  NDArray label = op::zeros(Shape(12, 1), nullptr, nullptr);
  AutoGrad ag(true, true);
  gluon::SoftmaxCrossEntropyLoss loss_fn;
  std::unordered_map<std::string, double> opt_params;
  opt_params["lr"] = 0.05;
  Trainer trainer(model.collect_parameters(), "sgd", opt_params);
  for (size_t epoch = 1; epoch <= 10; ++epoch) {
    ag.start_recording();
    NDArray ret = model(data);
    NDArray loss = loss_fn(ret, label);
    ag.backward(loss);
    ag.finish_recording();
    trainer.step();
    std::cout << loss.item<float>() << std::endl;
  }
  NDArray::WaitAll();
}

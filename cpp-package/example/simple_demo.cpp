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

class Model : public gluon::Block {
 public:
  Model() {
    dense0 = register_block("dense0", gluon::nn::Dense(64, 784, "relu"));
    dropout0 = register_block("dropout0", gluon::nn::Dropout(0.5));
    dense1 = register_block("dense1", gluon::nn::Dense(32, 64, "sigmoid"));
    dense2 = register_block("dense2", gluon::nn::Dense(10, 32));
  }

  NDArray forward(NDArray x) {
    x = dense0(x);
    x = dropout0(x);
    x = dense1(x);
    x = dense2(x);
    return x;
  }
  gluon::nn::Dense dense0, dense1, dense2;
  gluon::nn::Dropout dropout0;
};

int main(int argc, char** argv) {
  Model model;
  model.initialize<Uniform>(Uniform(0.07));
  int batch_size = 32;
  // NDArray data = op::ones(Shape(12, 784), nullptr, nullptr);
  AutoGrad ag(true, true);
  std::vector<std::string> data_files = { "./data/mnist_data/train-images-idx3-ubyte",
                                          "./data/mnist_data/train-labels-idx1-ubyte",
                                          "./data/mnist_data/t10k-images-idx3-ubyte",
                                          "./data/mnist_data/t10k-labels-idx1-ubyte"
                                        };
  auto train_iter = MXDataIter("MNISTIter");
  std::unordered_map<std::string, double> opt_params;
  opt_params["lr"] = 0.5;
  if (!setDataIter(&train_iter, "Train", data_files, batch_size)) {
    return 1;
  }
  Trainer trainer(model.collect_parameters(), "sgd", opt_params);
  gluon::SoftmaxCrossEntropyLoss loss_fn;
  for (size_t epoch = 1; epoch <= 100; ++epoch) {
    size_t batch_index = 0;
    train_iter.Reset();
    while (train_iter.Next()) {
      auto batch = train_iter.GetDataBatch();
      ag.start_recording();
      NDArray pred = model(batch.data);
      NDArray loss = loss_fn(pred, batch.label);
      ag.backward(loss);
      ag.finish_recording();
      trainer.step();
      NDArray::WaitAll();
      if (++batch_index % 100 == 0) {
        std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                  << " | Loss: " << loss.item<float>() << std::endl;
      }
    }
  }
}

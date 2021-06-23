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

MXNET_DEFINE_OP(add_scalar)
MXNET_DEFINE_OP(add)
MXNET_DEFINE_OP(softmax)
MXNET_DEFINE_OP(ones)
MXNET_DEFINE_OP(split)



// class Model : public gluon::Block {
//  public:
//   Model() {
//     dense0 = register_block("dense0", gluon::nn::Dense(10, 10));
//     dense1 = register_block("dense1", gluon::nn::Dense(10, 10));
//   }

//   NDArray forward(NDArray x) {
//     x = dense0(x);
//     x = dense1(x);
//     return x;
//   }

//   gluon::nn::Dense dense0, dense1;
// };

// using namespace mxnet::cpp;

int main(int argc, char** argv) {
  Context ctx_cpu = Context(DeviceType::kCPU, 0);
  NDArray data_array = NDArray({0.f, 0.3f, 0.6f, 0.9f}, Shape(4), ctx_cpu);
  // double result = op::add_scalar(3, 1, 1, 3.5, 2.52, 3, 0.51, 0.25);
  NDArray out = NDArray(Shape(4), ctx_cpu);
  NDArray result = op::add(data_array, 1, out);
  auto start = std::chrono::high_resolution_clock::now();
  NDArray result2 = op::activation(result, "sigmoid");
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << result2 << std::endl;
  std::cout << duration.count() << std::endl;

  NDArray input = NDArray({1.f, 2.f, 0.1f}, Shape(3), ctx_cpu);
  auto start2 = std::chrono::high_resolution_clock::now();
  NDArray result3 = op::log_softmax(input, 0, nullptr, false, "float32");
  auto stop2 = std::chrono::high_resolution_clock::now();
  auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
  std::cout << result3 << std::endl;
  std::cout << duration2.count() << std::endl;

  // auto start3 = std::chrono::high_resolution_clock::now();
  // NDArray ones_array = op::ones(Shape(2, 3, 1, 5), nullptr, nullptr);
  // auto stop3 = std::chrono::high_resolution_clock::now();
  // auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(stop3 - start3);
  // std::cout << ones_array << std::endl;
  // std::cout << duration3.count() << std::endl;

  NDArray ones_array = op::ones(Shape(2, 3, 1, 5), nullptr, nullptr);
  auto start3 = std::chrono::high_resolution_clock::now();
  std::vector<NDArray> split_arr = op::split(ones_array, 1, 0);
  auto stop3 = std::chrono::high_resolution_clock::now();
  auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(stop3 - start3);
  std::cout << split_arr[0] << std::endl;
  std::cout << duration3.count() << std::endl;

  NDArray neg = op::negative(ones_array, nullptr);
  std::cout << neg << std::endl;
}

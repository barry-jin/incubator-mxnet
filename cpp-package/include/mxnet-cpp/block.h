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
* \file block.h
* \brief definition of block
* \author Zhenghui Jin
*/

#ifndef MXNET_CPP_GLUON_BLOCK_H_
#define MXNET_CPP_GLUON_BLOCK_H_

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
#include "mxnet-cpp/symbol.h"
#include "mxnet-cpp/op_rt_map.h"
#include "mxnet-cpp/initializer.h"


namespace mxnet {
namespace cpp {
namespace gluon {

/*!
* \brief Block interface
*/
class Block {
 public:
  using paramType = std::vector<std::tuple<std::string, NDArray, NDArray, OpReqType>>;
  /*!
   * \brief constructor
   */
  Block() { this->need_cache = true; }
  /*!
   * \brief destructor
   */
  virtual ~Block() = default;
  /*!
   * \brief Register children blocks
   * \tparam BlockType The block type.
   * \param name name of the children block
   * \param block the pointer to the child block
   * \return reference to the child block
   */
  template <typename BlockType>
  BlockType register_block(std::string name, BlockType block);
  /*!
   * \brief Register parameters
   * \param name name of the children block
   * \param shape the parameter's shape
   * \param ctx the context of parameter NDArray, default as cpu()
   * \return parameter NDArray
   */
  NDArray register_parameter(std::string name, Shape shape, OpReqType grad_req = OpReqType::kWriteTo, Context ctx = Context::cpu());
  /*!
   * \brief Collect parameters
   * \param prefix prefix of the parameter name
   * \return updated parameters
   */
  paramType collect_parameters(std::string prefix = "");
  /*!
   * \brief Initialize parameters
   * \tparam InitType The Initializer type.
   * \param init initializer to initialize
   * \param ctx context, default to cpu
   */
  template <typename InitType>
  void initialize(InitType init, Context ctx = Context::cpu());
  /*!
   * \brief Initialize block
   * \param ctx context, default to cpu
   */
  void initialize(Context ctx = Context::cpu());
  /*!
   * \brief forward method
   * \param arr input NDArray
   * \return output NDArray
   */ 
  virtual NDArray forward(NDArray arr) {
    return hybrid_forward(arr);
  }
  /*!
   * \brief forward method
   * \param arr1 input first NDArray
   * \param arr2 input second NDArray
   * \return output NDArray
   */ 
  virtual NDArray forward(NDArray arr1, NDArray arr2) {
    return hybrid_forward(arr1, arr2);
  }
  /*!
   * \brief forward method
   * \param arr1 input first NDArray
   * \param arr2 input second NDArray
   * \param arr3 input third NDArray
   * \return output NDArray
   */ 
  virtual NDArray forward(NDArray arr1, NDArray arr2, NDArray arr3) {
    return hybrid_forward(arr1, arr2, arr3);
  }
  /*!
   * \brief hybrid_forward method
   * \param arr input NDArray
   * \return output NDArray
   */ 
  virtual NDArray hybrid_forward(NDArray arr) { return arr; }
  /*!
   * \brief hybrid_forward method
   * \param arr1 input first NDArray
   * \param arr2 input second NDArray
   * \return output NDArray
   */ 
  virtual NDArray hybrid_forward(NDArray arr1, NDArray arr2) { return arr1; }
  /*!
   * \brief hybrid_forward method
   * \param arr1 input first NDArray
   * \param arr2 input second NDArray
   * \param arr3 input third NDArray
   * \return output NDArray
   */ 
  virtual NDArray hybrid_forward(NDArray arr1, NDArray arr2, NDArray arr3) { return arr1; }
  /*!
   * \brief base case
   * \param arg input NDArray
   */ 
  inline void push_inputs(NDArray arg) {
    _input_args.push_back(arg);
  }
  /*!
   * \brief recursive case
   * \param arg input NDArray
   */ 
  template <typename... Args>
  inline void push_inputs(NDArray arg, Args... args) {
    _input_args.push_back(arg);
    push_inputs(args...);
  }
  /*!
   * \brief Call block with input NDArrays
   * \param args input NDArrays
   * \return output NDArray
   */
  template <typename... Args>
  NDArray operator()(Args... args) {
    _input_args.clear();
    push_inputs(args...);
    int curr;
    CHECK_EQ(MXNDArrayIsDeferredCompute(&curr), 0);
    if (curr) {
      if (_input_args.size() == 1) {
        return forward(_input_args[0]);
      } else if (_input_args.size() == 2) {
        return forward(_input_args[0], _input_args[1]);
      } else if (_input_args.size() == 3) {
        return forward(_input_args[0], _input_args[1], _input_args[2]);
      } else {
        LOG(FATAL) << "We only support input arguments size with 1, 2 or 3";
        return NDArray();
      }
    } else {
      if (need_cache) {
        return call_cached_op();
      } else {
        if (_input_args.size() == 1) {
          return forward(_input_args[0]);
        } else if (_input_args.size() == 2) {
          return forward(_input_args[0], _input_args[1]);
        } else if (_input_args.size() == 3) {
          return forward(_input_args[0], _input_args[1], _input_args[2]);
        } else {
          LOG(FATAL) << "We only support input arguments size with 1, 2 or 3";
          return NDArray();
        }        
      }
    }
  }
 protected:
  /*!
   * \brief Call CachedOp to create graph and invoke
   * \return output NDArray
   */
  NDArray call_cached_op() {
    if (this->_cached_op == nullptr) {
      build_cache();
    }
    std::vector<NDArrayHandle> arg_handles;
    for (const auto &array : arg_arrays) {
      arg_handles.push_back(array.GetHandle());
    }
    int prev_is_record = 0;
    int prev_train_mode = 0;
    CHECK_EQ(MXAutogradSetIsRecording(1, &prev_is_record), 0);
    CHECK_EQ(MXAutogradSetIsTraining(1, &prev_train_mode), 0);
    std::vector<NDArrayHandle> output_handles;
    std::transform(out_arrays.begin(), out_arrays.end(),
        std::back_inserter(output_handles), [](NDArray& a) {
          return a.GetHandle();
        });
    int out_size = 0;
    NDArrayHandle *out_array_handle = nullptr;
    // CHECK_EQ(MXInvokeCachedOp(_cached_op, arg_handles.size(), arg_handles.data(),
    //                           device_type, device_id, &out_size, &out_array_handle, nullptr),
    //          0);
    int err = MXInvokeCachedOp(_cached_op, arg_handles.size(), arg_handles.data(),
                               device_type, device_id, &out_size, &out_array_handle, nullptr);
    if (err != 0) {
      LOG(FATAL) << MXGetLastError();
    }
    out_arrays.clear();
    out_arrays.reserve(out_size);
    for (mx_uint i = 0; i < out_size; ++i) {
      out_arrays.push_back(NDArray(out_array_handle[i]));
    }
    int cur_train_mode = prev_train_mode;
    int cur_is_record = prev_is_record;
    CHECK_EQ(MXAutogradSetIsTraining(cur_train_mode, &prev_train_mode), 0);
    CHECK_EQ(MXAutogradSetIsRecording(cur_is_record, &prev_is_record), 0);
    return out_arrays.back();
  }
  /*!
   * \brief Initialize CachedOp
   */
  void build_cache() {
    std::unordered_map<std::string, index_t> index_map;
    paramType params = collect_parameters();
    std::vector<NDArrayHandle> real_params;
    std::vector<SymbolHandle> params_sym;
    std::vector<Symbol> temp_sym;
    real_params.reserve(params.size());
    params_sym.reserve(params.size());
    temp_sym.reserve(params.size());
    for (int i = 0; i < params.size(); i++) {
      real_params.emplace_back(std::get<1>(params[i]).GetHandle());
      temp_sym.emplace_back(Symbol::Variable(std::get<0>(params[i])));
      params_sym.emplace_back(temp_sym.back().GetHandle());
    }

    CHECK_EQ(MXNDArraySetDeferredComputeVariable(real_params.data(), params_sym.data(), real_params.size()), 0);
    std::vector<SymbolHandle> data;
    SymbolHandle out = get_graph(data);
    CHECK_EQ(MXCreateCachedOp(out, 0, nullptr, nullptr, &_cached_op, false), 0);
    Symbol* out_sym = new Symbol(out);
    std::vector<std::string> arg_name_list = out_sym->ListArguments();
    
    for (int i = 0; i < params.size(); i++) {
      index_map.emplace(std::get<0>(params[i]), i);
    }
    for (int i = 0; i < arg_name_list.size(); i++) {
      const auto &arg_name = arg_name_list[i];
      auto iter_arg = index_map.find(arg_name);
      if (iter_arg != index_map.end()) {
        arg_arrays.emplace_back(std::get<1>(params[iter_arg->second]));
      } else {
        auto iter_arg2 = _input_args_map.find(arg_name);
        if (iter_arg2 != _input_args_map.end()) {
          arg_arrays.emplace_back(iter_arg2->second);
        } else {
          LOG(FATAL) << "Unkonw argument name";
        }
      }
    }
  }
  /*!
   * \brief Get Cached Graph
   * \param data input symbol handles
   * \return output symbol handle
   */
  SymbolHandle get_graph(std::vector<SymbolHandle> &data) {
    std::vector<NDArrayHandle> real_args;
    std::vector<Symbol> temp_sym;
    NDArray out;
    if (_input_args.size() == 1) {
      temp_sym.emplace_back(Symbol::Variable("data"));
      data.emplace_back(temp_sym.back().GetHandle());
      real_args.emplace_back(_input_args[0].GetHandle());
      _input_args_map.emplace("data", _input_args[0]);
    } else {
      for (int i = 0; i < _input_args.size(); i++) {
        temp_sym.emplace_back(Symbol::Variable("data" + std::to_string(i)));
        data.emplace_back(temp_sym.back().GetHandle());
        real_args.emplace_back(_input_args[i].GetHandle());
        _input_args_map.emplace("data" + std::to_string(i), _input_args[0]);
      }
    }
    CHECK_EQ(MXNDArraySetDeferredComputeVariable(real_args.data(), data.data(), real_args.size()), 0);

    int prev_is_record;
    int prev_train_mode;
    int prev_is_deferred_compute;
    CHECK_EQ(MXAutogradSetIsRecording(0, &prev_is_record), 0);
    CHECK_EQ(MXAutogradSetIsTraining(0, &prev_train_mode), 0);
    CHECK_EQ(MXNDArraySetIsDeferredCompute(1, &prev_is_deferred_compute), 0);

    if (_input_args.size() == 1) {
      out = forward(_input_args[0]);
    } else if (_input_args.size() == 2) {
      out = forward(_input_args[0], _input_args[1]);
    } else if (_input_args.size() == 3) {
      out = forward(_input_args[0], _input_args[1], _input_args[2]);
    } else {
      LOG(FATAL) << "We only support input arguments size with 1, 2 or 3";
    }

    int cur_is_record = prev_is_record;
    int cur_train_mode = prev_train_mode;
    int cur_is_deferred_compute = prev_is_deferred_compute;
    CHECK_EQ(MXNDArraySetIsDeferredCompute(cur_is_deferred_compute, &prev_is_deferred_compute), 0);
    /*! prev_train_mode != enter_train_mode */
    if (prev_train_mode != 0) {
      CHECK_EQ(MXAutogradSetIsTraining(cur_train_mode, &prev_train_mode), 0);
    }
    /*! prev_is_record != enter_is_record */
    if (prev_is_record != 0) {
      CHECK_EQ(MXAutogradSetIsRecording(cur_is_record, &prev_is_record), 0);
    }
    SymbolHandle out_symbol;
    std::vector<NDArrayHandle> output_handles;
    output_handles.emplace_back(out.GetHandle());
    CHECK_EQ(MXNDArrayGetDeferredComputeSymbol(output_handles.data(), 1, &out_symbol), 0);
    return out_symbol;
  }
  /*! \brief need create cached_op */
  bool need_cache;
 private:
  /*! \brief Name-Block pair vector */
  std::vector<std::pair<std::string, std::shared_ptr<Block>>> _children_vec;
  /*! \brief Name-Param tuple vector */
  paramType _param_vec;
  /*! \brief vector storing input args */
  std::vector<NDArray> _input_args;
  /*! \brief mapping from names to args */
  std::unordered_map<std::string, NDArray> _input_args_map;
  /*! \brief cachedOp handle */
  CachedOpHandle _cached_op = nullptr;
  /*! \brief collected arg NDArray handles */
  std::vector<NDArray> arg_arrays;
  /*! \brief output NDArray handles */
  std::vector<NDArray> out_arrays;
  /*! \brief context device type */
  int device_type;
  /*! \brief context device id */
  int device_id;
};

}  // namespace gluon
}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_GLUON_BLOCK_H_

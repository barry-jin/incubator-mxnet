#include <iostream>

#include "mxnet/c_api.h"
#include "nnvm/c_api.h"

#define checkedMXCall(func, ...)                              \
  {                                                           \
    if (func(__VA_ARGS__) != 0) {                             \
      printf("MX call %s failed at line %d:\n%s",             \
            #func, __LINE__, MXGetLastError());               \
      exit(1)               ;                                 \
    }                                                         \
  }

int main() {

  /* Create symbol */
  const char json[] = "{\"nodes\": [{\"op\":\"null\",\"name\":\".Inputs.Input1\",\"inputs\":[]},{\"op\":\"null\",\"name\":\".Inputs.Input2\",\"inputs\":[]},{\"op\":\"elemwise_add\",\"name\":\".$0\",\"inputs\":[[0,0,0],[1,0,0]]},{\"op\":\"_copy\",\"name\":\".Outputs.Output\",\"inputs\":[[2,0,0]]}],\"arg_nodes\":[0,1],\"heads\":[[3,0,0]]}";
  SymbolHandle sym;
  checkedMXCall(MXSymbolCreateFromJSON, json, &sym);

  /* Create NDArray for arguments */
  int dev_type = 1;
  int dev_id = 0; 
  mx_uint shape[1] = {1};
  void *data;
  NDArrayHandle in_arr_1, in_arr_2;
  checkedMXCall(MXNDArrayCreate, shape, 1, dev_type, dev_id, 0, 0, &in_arr_1);
  checkedMXCall(MXNDArrayCreate, shape, 1, dev_type, dev_id, 0, 0, &in_arr_2);
  checkedMXCall(MXNDArrayGetData, in_arr_1, &data);
  /* Set values for arguments */
  *reinterpret_cast<float*>(data) = 0.4;
  checkedMXCall(MXNDArrayGetData, in_arr_2, &data);
  *reinterpret_cast<float*>(data) = 0.5;

  /* Create NDArray for gradients */
  NDArrayHandle grad_arr_1, grad_arr_2;
  checkedMXCall(MXNDArrayCreate, shape, 1, dev_type, dev_id, 0, 0, &grad_arr_1);
  checkedMXCall(MXNDArrayCreate, shape, 1, dev_type, dev_id, 0, 0, &grad_arr_2);
  /* Set values for gradients */
  checkedMXCall(MXNDArrayGetData, grad_arr_1, &data);
  *reinterpret_cast<float*>(data) = 0;
  checkedMXCall(MXNDArrayGetData, grad_arr_2, &data);
  *reinterpret_cast<float*>(data) = 0;
  /* Attach gradients to arguments */
  uint32_t grad_req_1[1] = {1};
  uint32_t grad_req_2[1] = {1}; 
  checkedMXCall(MXAutogradMarkVariables, 1, &in_arr_1, grad_req_1, &grad_arr_1);
  checkedMXCall(MXAutogradMarkVariables, 1, &in_arr_2, grad_req_2, &grad_arr_2);

  /* Create cached op */
  const char *cachedop_keys[1] = {"static_alloc"};
  const char *cachedop_vals[1] = {"true"};
  CachedOpHandle cached_op;
  checkedMXCall(MXCreateCachedOp, sym, 1, cachedop_keys, cachedop_vals, &cached_op, false);

  /* Set autograd to record & set training mode */
  int dummy_prev;
  checkedMXCall(MXAutogradSetIsRecording, 1, &dummy_prev);
  checkedMXCall(MXAutogradSetIsTraining, 1, &dummy_prev);

  /* Run forward */
  int n_outs;
  NDArrayHandle *out_arr_p = nullptr;
  const int *dummy_stypes = nullptr;
  NDArrayHandle inputs[] = {in_arr_1, in_arr_2};
  checkedMXCall(MXInvokeCachedOp, cached_op, 2, inputs, dev_type, dev_id, &n_outs, &out_arr_p, &dummy_stypes);
  checkedMXCall(MXNDArrayWaitToRead, *out_arr_p);

  /* Create NDArray for outgrad */
  NDArrayHandle outgrad_arr;
  checkedMXCall(MXNDArrayCreate, shape, 1, dev_type, dev_id, 0, 0, &outgrad_arr);
  /* Set values for outgrad */
  checkedMXCall(MXNDArrayGetData, outgrad_arr, &data);
  *reinterpret_cast<float*>(data) = 1;

  /* Run backward */
  checkedMXCall(MXAutogradBackward, 1, out_arr_p, &outgrad_arr, false);
  checkedMXCall(MXNDArrayWaitToRead, grad_arr_1);
  checkedMXCall(MXNDArrayWaitToRead, grad_arr_2);

  /* Check results */
  checkedMXCall(MXNDArrayGetData, in_arr_1, &data);
  std::cout << "INPUT 1: " << *reinterpret_cast<float*>(data) << "\n"; 
  checkedMXCall(MXNDArrayGetData, in_arr_2, &data);
  std::cout << "INPUT 2: " << *reinterpret_cast<float*>(data) << "\n"; 
  checkedMXCall(MXNDArrayGetData, *out_arr_p, &data);
  std::cout << "OUTPUT: " << *reinterpret_cast<float*>(data) << "\n"; 
  checkedMXCall(MXNDArrayGetData, outgrad_arr, &data);
  std::cout << "OUTGRAD: " << *reinterpret_cast<float*>(data) << "\n"; 
  checkedMXCall(MXNDArrayGetData, grad_arr_1, &data);
  std::cout << "GRAD 1: " << *reinterpret_cast<float*>(data) << "\n"; 
  checkedMXCall(MXNDArrayGetData, grad_arr_2, &data);
  std::cout << "GRAD 2: " << *reinterpret_cast<float*>(data) << "\n"; 

  return 0;
}
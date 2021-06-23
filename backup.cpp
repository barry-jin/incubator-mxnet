static OpMapNew *op_map_ = new OpMapNew();
MXNetFunctionHandle _handle = op_map_->GetOpHandle("log_softmax");

inline NDArray log_softmax(NDArray input, int axis, nullptr_t temperature, bool use_length, const char* dtype) {
  std::vector<MXNetValue> arg_values;
  std::vector<int> type_codes;
  MXNetValue* _value = new MXNetValue();
  int ret_type_code;
  MXNetValue* temp1 = new MXNetValue();
  MXNetValue* temp2 = new MXNetValue();
  MXNetValue* temp3 = new MXNetValue();
  MXNetValue* temp4 = new MXNetValue();
  MXNetValue* temp5 = new MXNetValue();
  temp1->v_handle = input.GetHandle();
  temp2->v_int64 = axis;
  temp3->v_handle = nullptr;
  temp4->v_int64 = use_length;
  temp5->v_str = dtype;
  arg_values.push_back(*temp1);
  arg_values.push_back(*temp2);
  arg_values.push_back(*temp3);
  arg_values.push_back(*temp4);
  arg_values.push_back(*temp5);
  type_codes.push_back(kNDArrayHandle);
  type_codes.push_back(kDLInt);
  type_codes.push_back(kNull);
  type_codes.push_back(kDLInt);
  type_codes.push_back(kStr);
  CHECK_EQ(MXNetFuncCall(_handle, arg_values.data(), type_codes.data(), type_codes.size(), _value, &ret_type_code), 0);
  NDArray* ret = new NDArray(_value->v_handle);
  return *ret;
}

  PreatyPrint(out, shape.size()-2, shape, 0, 0, cpu_array.GetData());

// inline void PreatyPrint(std::ostream &out, int depth, std::vector<mx_uint> shape,
//                         int round, int offsite, const mx_float* data) {
//   if (depth >= 0) {
//     int num_fields = shape[shape.size()-1];
//     if (round == 0) {
//       out << '[';
//     }
//     out << '[';
//     std::copy(data + offsite, data + offsite + num_fields,
//           std::ostream_iterator<float>(out, ", "));
//     if (round == shape[depth]) {
//       out << "]]";
//     } else {
//       out << "],";
//     }
//     if (round == shape[depth]) {
//       PreatyPrint(out, depth-1, shape, 0, offsite + num_fields, data);
//     } else {
//       PreatyPrint(out, depth, shape, round+1, offsite + num_fields, data);
//     }
//   }
// }

  /*! \brief Mapping from Name to the index of the Block stored */
  std::unordered_map<std::string, index_t> _param_map;
    /*! \brief Mapping from Name to the index of the Block stored */
  std::unordered_map<std::string, index_t> _children_map;

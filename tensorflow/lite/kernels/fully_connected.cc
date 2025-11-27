/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/kernels/internal/optimized/integer_ops/fully_connected.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/optimized/fully_connected_4bit.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/optimized/sparse_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/reference/sparse_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/minimal_logging.h"

#ifdef TFLITE_HAVE_CPUINFO
#include "include/cpuinfo.h"
#endif

#if defined(__APPLE__) || defined(__linux__) || defined(__Fuchsia__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "tensorflow/lite/kernels/FullPack/low_precision_fully_connected.h"
#include "tensorflow/lite/kernels/FullPack/common/types.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace fully_connected {

namespace {
bool SupportedSparsityFormat(const TfLiteSparsity& sparsity) {
  if (sparsity.dim_metadata[0].format == kTfLiteDimDense &&
      sparsity.dim_metadata[1].format == kTfLiteDimSparseCSR) {
    return true;
  }

  return false;
}

static const int kDimMetadataSizeRandomSparse = 2;
static const int kDimMetadataSizeBlockSparse = 3;

TfLiteStatus CreateLedgerTensor(const TfLiteSparsity* sparsity,
                                TfLiteContext* context, TfLiteTensor* ledger) {
  TF_LITE_ENSURE(context, sparsity != nullptr);
  ledger->name = "FC_ledger";
  ledger->type = kTfLiteUInt8;
  ledger->allocation_type = kTfLiteArenaRwPersistent;
  TfLiteIntArray* ledger_size = TfLiteIntArrayCreate(1);
  ledger_size->data[0] = sparsity->dim_metadata[1].array_indices->size +
                         sparsity->dim_metadata[1].array_segments->size - 1;
  return context->ResizeTensor(context, ledger, ledger_size);
}

TfLiteStatus PopulateLedgerData(const TfLiteSparsity* sparsity,
                                TfLiteContext* context, uint8_t* ledger_data) {
  TF_LITE_ENSURE(context, sparsity != nullptr);
  const auto* array_segments = sparsity->dim_metadata[1].array_segments;
  const auto* array_indices = sparsity->dim_metadata[1].array_indices;
  int output_data_ptr = 0;

  for (int i = 0; i < array_segments->size - 1; i++) {
    int row_start = array_segments->data[i];
    int row_end = array_segments->data[i + 1];
    if (row_end - row_start > UINT8_MAX) {
      return kTfLiteError;
    }
    // Copy num of non-zero blocks in row i.
    ledger_data[output_data_ptr] = static_cast<uint8_t>(row_end - row_start);
    output_data_ptr++;

    for (int j = row_start; j < row_end; j++) {
      if (array_indices->data[j] > UINT8_MAX) {
        return kTfLiteError;
      }
      // Copy indices of non-zero blocks in row i.
      ledger_data[output_data_ptr] =
          static_cast<uint8_t>(array_indices->data[j]);
      output_data_ptr++;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus VerifyPerChannelQuantization(TfLiteContext* context,
                                          const TfLiteTensor* tensor) {
  TF_LITE_ENSURE_EQ(context, tensor->quantization.type,
                    kTfLiteAffineQuantization);
  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(tensor->quantization.params);
  TF_LITE_ENSURE(context, affine_quantization);
  TF_LITE_ENSURE(context, affine_quantization->scale);
  return affine_quantization->scale->size > 1 ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus VerifyQuantizationZeroPoint(const TfLiteTensor* tensor,
                                         int expected_value) {
  if (tensor->quantization.type == kTfLiteAffineQuantization) {
    const auto* params = reinterpret_cast<TfLiteAffineQuantization*>(
        tensor->quantization.params);
    if (params && params->zero_point &&
        std::any_of(params->zero_point->data,
                    params->zero_point->data + params->zero_point->size,
                    [expected_value](int v) { return v != expected_value; })) {
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace

// This file has four implementations of FullyConnected
enum KernelType {
  kReference,
  kGenericOptimized,
  kLegacyPie,  // Legacy path used by the PIE team and related clients.
};

struct OpData {
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  // Per channel output multiplier and shift.
  std::vector<int32_t> per_channel_output_multiplier;
  std::vector<int> per_channel_output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
  // The index of the temporary tensor where the quantized inputs are cached.
  int scratch_tensor_index;
  bool compute_row_sums = false;
  // Only used for sparse hybrid fully connected kernels.
  bool ledger_initialized;
  // Used for 4bit hybrid
  std::unique_ptr<optimized_4bit::OpData4Bit> op_data_4bit = nullptr;
  TfLiteType quantized_bias_type = kTfLiteNoType;
  // LowPrecision
  bool low_precision_applicable = false;
  bool low_precision_multibatched = false;
  bool low_precision_compress_activation = false;
  LowPrecision::Method operation_method = LowPrecision::Method::kNoOptimization;
  long int low_precision_id = 0;
  int kernel_temps;
  int input_temps;
  int output_temps;
  int kernel_temps_idx;
  int input_temps_idx;
  int output_temps_idx;
  int filter_temps_idx;
  LowPrecision::Matrix* filter_matrix = nullptr;
  LowPrecision::TimingDetailes* timing_details = nullptr;
};

constexpr int kInputTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;
constexpr int kShuffledInputWorkspaceTensor = 1;

// Begin temporary tensor ids created at init and initialized during prepare.
constexpr int kQuantizedInputTensor = 0;
constexpr int kScalingFactorsTensor = 1;
constexpr int kAccumulatorTensor = 2;
constexpr int kInputOffsetsTensor = 3;

inline TfLiteStatus CheckTypes(TfLiteContext* context,
                               const TfLiteTensor* input,
                               const TfLiteTensor* filter,
                               const TfLiteTensor* bias, TfLiteTensor* output,
                               TfLiteFullyConnectedParams* params) {
  const bool is_quantized =
      ((filter->type == kTfLiteUInt8) || (filter->type == kTfLiteInt8) ||
       (filter->type == kTfLiteInt4));
  const bool is_hybrid = is_quantized && (input->type == kTfLiteFloat32);
  const bool is_shuffled =
      is_quantized && (params->weights_format ==
                       kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8);

  // optional bias tensor.
  const bool is_optional_bias_float = !bias || (bias->type == kTfLiteFloat32);
  const bool is_optional_bias_int =
      !bias || (bias->type == kTfLiteInt32) || (bias->type == kTfLiteInt64);

  if (is_quantized) {
    if (is_shuffled) {
      TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteUInt8);
      TF_LITE_ENSURE_TYPES_EQ(context, filter->type, kTfLiteUInt8);
      TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt16);
      TF_LITE_ENSURE_EQ(context, is_optional_bias_int, true);
    } else if (is_hybrid) {
      TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
      TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat32);
      TF_LITE_ENSURE_EQ(context, is_optional_bias_float, true);
    } else {
      TF_LITE_ENSURE(context, input->type == kTfLiteUInt8 ||
                                  input->type == kTfLiteInt8 ||
                                  input->type == kTfLiteInt16);
      TF_LITE_ENSURE(context, output->type == kTfLiteUInt8 ||
                                  output->type == kTfLiteInt8 ||
                                  output->type == kTfLiteInt16);
      TF_LITE_ENSURE_EQ(context, is_optional_bias_int, true);
    }
  } else {
    // Only float32 is supported currently
    TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat32);
    TF_LITE_ENSURE_TYPES_EQ(context, filter->type, kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, is_optional_bias_float, true);
  }

  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
#ifdef TFLITE_HAVE_CPUINFO
  // We ensure that cpuinfo is initialized to correctly detect the optimized
  // paths we can take. Note the we do not call `cpuinfo_deinitialize` in
  // `Free`: that operation is currently a no-op AND we want to avoid
  // deinitializing cpuinfo for other parts of the program that could need it
  // after we free the op if it ever does perform something.
  if (!cpuinfo_initialize()) {
    TFLITE_LOG(tflite::TFLITE_LOG_WARNING,
               "Could not initialize cpuinfo, some optimization opportunities "
               "may be missed.");
  }
#endif
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().
  auto* op_data = new OpData();
  op_data->scratch_tensor_index = -1;
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  OpData* data = reinterpret_cast<OpData*>(buffer);
  if (data->low_precision_applicable){
    // std::cout << "GEMM API Timing (FC-" << data->low_precision_id 
    //           << "): " << data->timing_details->total() * 1000000 << std::endl;
    // std::cout << "\t" << "GEMM            : " << data->timing_details->gemm * 1000000 << std::endl;
    // std::cout << "\t" << "Output UnPacking: " << data->timing_details->packing * 1000000 << std::endl;
    // std::cout << "\t" << "Output DePadding: " << data->timing_details->padding * 1000000 << std::endl;
    delete data->filter_matrix;
    // delete data->timing_details;
    LowPrecision::timingManager.addTimingDetail(data->timing_details);
  }
  delete data;
}

/*TfLiteStatus PrepareImpl(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  // Check we have all the inputs and outputs we need.
  TF_LITE_ENSURE(context, node->inputs->size == 2 || node->inputs->size == 3);
  // Shuffled formats need a workspace to store the shuffled input activations.
  const int expected_outputs_count =
      params->weights_format == kTfLiteFullyConnectedWeightsFormatDefault ? 1
                                                                          : 2;
  TF_LITE_ENSURE_EQ(context, node->outputs->size, expected_outputs_count);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* filter;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kWeightsTensor, &filter));
  const TfLiteTensor* bias =
      (node->inputs->size == 3)
          ? GetOptionalInputTensor(context, node, kBiasTensor)
          : nullptr;
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // Check proper datatype match among all Input Tensors
  TF_LITE_ENSURE_STATUS(
      CheckTypes(context, input, filter, bias, output, params));

  // Check all the parameters of tensor match within themselves and match the
  // input configuration.
  int input_size = 1;
  for (int i = 0; i < input->dims->size; i++) {
    input_size *= input->dims->data[i];
  }

  TF_LITE_ENSURE_EQ(context, NumDimensions(filter), 2);
  TF_LITE_ENSURE(context, filter->dims->data[1] != 0);
  const int batch_size = input_size / filter->dims->data[1];
  const int num_units = filter->dims->data[0];

  if (bias) {
    TF_LITE_ENSURE_EQ(context, NumElements(bias), SizeOfDimension(filter, 0));
  }

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8 ||
      input->type == kTfLiteInt16) {
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, filter, bias, output, &real_multiplier));
    int exponent;
    QuantizeMultiplier(real_multiplier, &data->output_multiplier, &exponent);
    data->output_shift = exponent;
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, params->activation, output, &data->output_activation_min,
        &data->output_activation_max));
  }

  if (input->type == kTfLiteInt16 && output->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
  }

  // If we have to perform on-the-fly quantization (with quantized weights and
  // float inputs) first we need to quantize the inputs. Allocate a temporary
  // buffer to store the intermediate quantized values.
  // Additionally, we allocate a temporary buffer to store the accumulated
  // quantized values prior to multiplication by the scaling factor.
  const bool is_hybrid =
      (input->type == kTfLiteFloat32 &&
       (filter->type == kTfLiteUInt8 || filter->type == kTfLiteInt8));
  const bool is_sparse = filter->sparsity != nullptr;

  
  LowPrecision::Method __method = LowPrecision::FullyConnected::GetMethodFromEnv();
  int __dims = input->dims->size;
  int* __sizes = new int[__dims];
  for (int i = 0; i < input->dims->size; i++)
    __sizes[i] = input->dims->data[i];

  LowPrecision::Shape __shape = LowPrecision::get_shape(__sizes, __dims);

  int __filter_dims = 2;
  int __filter_sizes[2] = {filter->dims->data[0], filter->dims->data[1]};
  LowPrecision::Shape __filter_shape = LowPrecision::get_shape(__filter_sizes, __filter_dims);

  bool should_apply_low_precision = LowPrecision::FullyConnected::IsAppliable(
    __method, __shape, __filter_shape,
    LowPrecision::FullyConnected::GetDataType(input->type),
    LowPrecision::FullyConnected::GetDataType(filter->type),
    LowPrecision::FullyConnected::GetDataType(output->type),
    __shape.flatsize >= 2 * 2048
  );
  // std::cerr << "Method is 0x" << std::hex << ((int)__method) << std::dec << " and isApplicable is " << ((int)should_apply_low_precision) << std::endl;
  bool includes_low_precision_activation = LowPrecision::FullyConnected::IncludesActivationCompression(__method);
  if (LowPrecision::FullyConnected::GetVariableFromEnv( "ForceCaching" ) == "TRUE")
    CpuBackendContext::GetFromContext(context)->SetUseCaching(true);

  int qk_id = 0;
  int ip_id = 1;
  // int ca_id = 2;
  int num_tmps = (should_apply_low_precision)?((includes_low_precision_activation || batch_size > 1)?(7):(6)):(5);

  if (is_hybrid) {
    qk_id = 5;
    ip_id = 6;
    // ca_id = 7;
    TfLiteIntArrayFree(node->temporaries);
    data->compute_row_sums = true;
    if (is_sparse) {
      node->temporaries = TfLiteIntArrayCreate(num_tmps + 1);
    } else {
      node->temporaries = TfLiteIntArrayCreate(num_tmps);
    }
    node->temporaries->data[0] = data->scratch_tensor_index;

    TfLiteTensor* input_quantized;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, 0,
                                                &input_quantized));
    input_quantized->type = filter->type;
    input_quantized->allocation_type = kTfLiteArenaRw;

    TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                     input_quantized_size));

    node->temporaries->data[1] = data->scratch_tensor_index + 1;
    TfLiteTensor* scaling_factors;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node,1,
                                                &scaling_factors));
    scaling_factors->type = kTfLiteFloat32;
    scaling_factors->allocation_type = kTfLiteArenaRw;

    int scaling_dims[1] = {batch_size};
    if (!TfLiteIntArrayEqualsArray(scaling_factors->dims, 1, scaling_dims)) {
      TfLiteIntArray* scaling_factors_size = TfLiteIntArrayCreate(1);
      scaling_factors_size->data[0] = batch_size;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scaling_factors,
                                                       scaling_factors_size));
    }

    node->temporaries->data[2] = data->scratch_tensor_index + 2;
    TfLiteTensor* accum_scratch;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, 2, &accum_scratch));
    accum_scratch->type = kTfLiteInt32;
    accum_scratch->allocation_type = kTfLiteArenaRw;
    int accum_scratch_dims[2] = {num_units, batch_size};
    if (!TfLiteIntArrayEqualsArray(accum_scratch->dims, 2,
                                   accum_scratch_dims)) {
      TfLiteIntArray* accum_size = TfLiteIntArrayCreate(2);
      accum_size->data[0] = num_units;
      accum_size->data[1] = batch_size;
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, accum_scratch, accum_size));
    }

    node->temporaries->data[3] = data->scratch_tensor_index + 3;
    TfLiteTensor* input_offsets;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, 3, &input_offsets));
    input_offsets->type = kTfLiteInt32;
    input_offsets->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqualsArray(input_offsets->dims, 1, scaling_dims)) {
      TfLiteIntArray* input_offsets_size = TfLiteIntArrayCreate(1);
      input_offsets_size->data[0] = batch_size;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_offsets,
                                                       input_offsets_size));
    }
    node->temporaries->data[4] = data->scratch_tensor_index + 4;
    TfLiteTensor* row_sums;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, 4, &row_sums));
    row_sums->type = kTfLiteInt32;
    row_sums->allocation_type = kTfLiteArenaRwPersistent;
    int row_sums_dims[1] = {num_units};
    if (!TfLiteIntArrayEqualsArray(row_sums->dims, 1, row_sums_dims)) {
      TfLiteIntArray* row_sums_size = TfLiteIntArrayCreate(1);
      row_sums_size->data[0] = row_sums_dims[0];
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, row_sums, row_sums_size));
    }

    if (is_sparse) {
      data->ledger_initialized = false;
      node->temporaries->data[5] = data->scratch_tensor_index + 5;
      TfLiteTensor* filter_ledger =
          &context->tensors[node->temporaries->data[5]];
      auto status =
          CreateLedgerTensor(filter->sparsity, context, filter_ledger);
      if (status != kTfLiteOk) return status;
    }
  }
  else if (input->type == kTfLiteInt8){
    node->temporaries = TfLiteIntArrayCreate(num_tmps - 5);
  }

  if (should_apply_low_precision)
    std::cerr << "Applying Low-Precision for shape " 
              << LowPrecision::get_shape_string(__filter_shape) 
              << " and Input shape "
              << LowPrecision::get_shape_string(__shape)
              << " With " << num_tmps << " Number of Temporaries Tensors"
              << std::endl;
  else
    std::cerr << "NOT Applying Low-Precision for shape " 
              << LowPrecision::get_shape_string(__filter_shape) 
              << " and Input shape "
              << LowPrecision::get_shape_string(__shape)
              << " With " << num_tmps << " Number of Temp Tensors"
              << std::endl;

  if (should_apply_low_precision){
    data->operation_method = __method;
    LowPrecision::FullyConnected::set_default_method(__method);
    data->low_precision_applicable = true;
    node->temporaries->data[qk_id] = data->scratch_tensor_index + qk_id;
    TfLiteTensor* quantized_kernels = GetTemporary(context, node, qk_id);
    quantized_kernels->type = kTfLiteInt8;
    quantized_kernels->allocation_type = kTfLitePersistentRo;
    int quantized_kernels_dims[2] = {filter->dims->data[0], filter->dims->data[1]};
    LowPrecision::FullyConnected::TransformFilterShape(__method, quantized_kernels_dims, 2);
    std::cout << "\tTransformed Filter Shape: (" 
              << quantized_kernels_dims[0] << ", " << quantized_kernels_dims[1]
              << ")\n";
    if (!TfLiteIntArrayEqualsArray(quantized_kernels->dims, 2, quantized_kernels_dims)) {
      TfLiteIntArray* quantized_kernels_size = TfLiteIntArrayCreate(2);
      quantized_kernels_size->data[0] = quantized_kernels_dims[0];
      quantized_kernels_size->data[1] = quantized_kernels_dims[1];
      int* _sizes = new int[2];
      _sizes[0] = filter->dims->data[0];
      _sizes[1] = filter->dims->data[1];
      LowPrecision::Shape _shape = LowPrecision::get_shape(_sizes, 2);
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, quantized_kernels, quantized_kernels_size));
      LowPrecision::Status ret;
      ret = LowPrecision::FullyConnected::QuantizeFilter(
        __method, GetTensorData<int8_t>(filter),
        _shape, GetTensorData<int8_t>(quantized_kernels),
        LowPrecision::MemLayout::kRowMajor
      );
      TF_LITE_ASSERT_EQ(ret, LowPrecision::Status::Success);
      if (includes_low_precision_activation || batch_size > 1){
        data->low_precision_compress_activation = true;
        node->temporaries->data[ip_id] = data->scratch_tensor_index + ip_id;

        TfLiteTensor* input_compressed = GetTemporary(context, node, ip_id);
        input_compressed->type = filter->type;
        input_compressed->allocation_type = kTfLiteArenaRw;
        std::cout << "\tTransformed Activation Shape From: ("
                  << __sizes[0] << ", " << __sizes[1]
                  << ") ";
        LowPrecision::FullyConnected::TransformInputShape(__method, __sizes, 2);
        std::cout << "To: ("
                  << __sizes[0] << ", " << __sizes[1]
                  << ")\n";

        TfLiteIntArray* compressed_activation_size = TfLiteIntArrayCreate(2);
        compressed_activation_size->data[0] = __sizes[0];
        compressed_activation_size->data[1] = __sizes[1];
        TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_compressed,
                                                        compressed_activation_size));
      }
    }
  }

  // Resize output.
  TfLiteIntArray* output_size_array = nullptr;
  if (params->keep_num_dims) {
    // When number of dimensions are kept the filter operates along the last
    // dimensions. In other words, for an input tensor with shape
    // [batch_size, ..., n_inputs] and a filter of shape [n_inputs, n_units]
    // this Op produces an output of shape [batch_size, ..., n_units].
    TF_LITE_ENSURE_EQ(context, input->dims->data[input->dims->size - 1],
                      SizeOfDimension(filter, 1));
    output_size_array = TfLiteIntArrayCopy(input->dims);
    output_size_array->data[output_size_array->size - 1] = num_units;
  } else {
    // Otherwise, the output is (potentially flattened to) a 2-D matrix.
    output_size_array = TfLiteIntArrayCreate(2);
    output_size_array->data[0] = batch_size;
    output_size_array->data[1] = num_units;
  }
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size_array));

  return kTfLiteOk;
}*/

TfLiteStatus UpdateOutputSize(TfLiteContext* context,
                              TfLiteFullyConnectedParams* params,
                              const TfLiteTensor* input, TfLiteTensor* output,
                              int batch_size, int num_units, int cols) {
  TfLiteIntArray* output_size_array = nullptr;
  if (params->keep_num_dims) {
    TF_LITE_ENSURE_EQ(context, input->dims->data[input->dims->size - 1], cols);
    output_size_array = TfLiteIntArrayCopy(input->dims);
    output_size_array->data[output_size_array->size - 1] = num_units;
  } else {
    // Otherwise, the output is (potentially flattened to) a 2-D matrix.
    output_size_array = TfLiteIntArrayCreate(2);
    output_size_array->data[0] = batch_size;
    output_size_array->data[1] = num_units;
  }
  return context->ResizeTensor(context, output, output_size_array);
}

TfLiteStatus PrepareImpl4Bit(TfLiteContext* context, TfLiteNode* node,
                             int lhs_width, int rhs_width, int depth,
                             int batch_size, int cols, int output_depth) {
  const int units = output_depth;
  const int lhs_layout_cols =
      (cols + (optimized_4bit::FilterDepth - 1)) & ~(depth - 1);
  const int rhs_layout_rows = (batch_size + (rhs_width - 1)) & ~(rhs_width - 1);
  const int rhs_layout_cols = lhs_layout_cols;
  const int dst_layout_rows = rhs_layout_rows;
  const int dst_layout_cols = (units + (lhs_width - 1)) & ~(lhs_width - 1);

  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(5);
  for (int i = 0; i < 5; i++) {
    node->temporaries->data[i] = data->scratch_tensor_index + i;
  }

  TfLiteTensor* input_quantized;
  TF_LITE_ENSURE_OK(
      context,
      GetTemporarySafe(context, node, kQuantizedInputTensor, &input_quantized));
  input_quantized->type = kTfLiteInt8;
  input_quantized->allocation_type = kTfLiteArenaRw;
  const int input_quantized_dims[2] = {rhs_layout_rows, rhs_layout_cols};
  if (!TfLiteIntArrayEqualsArray(input_quantized->dims, 2,
                                 input_quantized_dims)) {
    TfLiteIntArray* input_quantized_size = TfLiteIntArrayCreate(2);
    input_quantized_size->data[0] = input_quantized_dims[0];
    input_quantized_size->data[1] = input_quantized_dims[1];
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                     input_quantized_size));
  }
  TfLiteTensor* scaling_factors;
  TF_LITE_ENSURE_OK(
      context,
      GetTemporarySafe(context, node, kScalingFactorsTensor, &scaling_factors));
  scaling_factors->type = kTfLiteFloat32;
  scaling_factors->allocation_type = kTfLiteArenaRw;
  const int scaling_factors_dims[1] = {rhs_layout_rows};
  if (!TfLiteIntArrayEqualsArray(scaling_factors->dims, 1,
                                 scaling_factors_dims)) {
    TfLiteIntArray* scaling_factors_size = TfLiteIntArrayCreate(1);
    scaling_factors_size->data[0] = scaling_factors_dims[0];
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scaling_factors,
                                                     scaling_factors_size));
  }

  TfLiteTensor* accum_scratch;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, kAccumulatorTensor,
                                              &accum_scratch));
  accum_scratch->type = kTfLiteInt32;
  accum_scratch->allocation_type = kTfLiteArenaRw;
  const int accum_scratch_dims[2] = {dst_layout_rows, dst_layout_cols};
  if (!TfLiteIntArrayEqualsArray(accum_scratch->dims, 2, accum_scratch_dims)) {
    TfLiteIntArray* accum_size = TfLiteIntArrayCreate(2);
    accum_size->data[0] = accum_scratch_dims[0];
    accum_size->data[1] = accum_scratch_dims[1];
    TF_LITE_ENSURE_OK(
        context, context->ResizeTensor(context, accum_scratch, accum_size));
  }

  TfLiteTensor* input_offsets;
  TF_LITE_ENSURE_OK(
      context,
      GetTemporarySafe(context, node, kInputOffsetsTensor, &input_offsets));
  input_offsets->type = kTfLiteInt32;
  input_offsets->allocation_type = kTfLiteArenaRw;
  const int input_offsets_dims[1] = {rhs_layout_rows};
  if (!TfLiteIntArrayEqualsArray(input_offsets->dims, 1, input_offsets_dims)) {
    TfLiteIntArray* input_offsets_size = TfLiteIntArrayCreate(1);
    input_offsets_size->data[0] = input_offsets_dims[0];
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_offsets,
                                                     input_offsets_size));
  }

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  auto* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
  return UpdateOutputSize(context, params, input, output, batch_size, units,
                          cols);
}

TfLiteStatus PrepareImpl(TfLiteContext* context, TfLiteNode* node,
                         KernelType kernel_type) {
  auto* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  // Check we have all the inputs and outputs we need.
  TF_LITE_ENSURE(context, node->inputs->size == 2 || node->inputs->size == 3);
  // Shuffled formats need a workspace to store the shuffled input activations.
  const int expected_outputs_count =
      params->weights_format == kTfLiteFullyConnectedWeightsFormatDefault ? 1
                                                                          : 2;
  TF_LITE_ENSURE_EQ(context, node->outputs->size, expected_outputs_count);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* filter;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kWeightsTensor, &filter));
  const TfLiteTensor* bias =
      (node->inputs->size == 3)
          ? GetOptionalInputTensor(context, node, kBiasTensor)
          : nullptr;
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // Check proper datatype match among all Input Tensors
  TF_LITE_ENSURE_STATUS(
      CheckTypes(context, input, filter, bias, output, params));

  // Check all the parameters of tensor match within themselves and match the
  // input configuration.
  int input_size = 1;
  for (int i = 0; i < input->dims->size; i++) {
    input_size *= input->dims->data[i];
  }

  TF_LITE_ENSURE_EQ(context, NumDimensions(filter), 2);

  // When the second dimension size of the filter tensor is 0, we need to
  // generate the output shape early to avoid dividing by 0.
  if (filter->dims->data[1] == 0) {
    TfLiteIntArray* output_size_array;
    if (params->keep_num_dims) {
      output_size_array = TfLiteIntArrayCopy(input->dims);
      output_size_array->data[output_size_array->size - 1] =
          filter->dims->data[0];
    } else {
      output_size_array = TfLiteIntArrayCreate(2);
      // If `keep_num_dims` is false, we need to flatten the output tensor to
      // have rank 2.
      int batch_size = 1;
      for (int i = 0; i < input->dims->size - 1; ++i)
        batch_size *= input->dims->data[i];
      output_size_array->data[0] = batch_size;
      output_size_array->data[1] = filter->dims->data[0];
    }
    TF_LITE_ENSURE_OK(
        context, context->ResizeTensor(context, output, output_size_array));
    return kTfLiteOk;
  }

  const int batch_size = input_size / filter->dims->data[1];
  const int num_units = filter->dims->data[0];

  if (bias) {
    TF_LITE_ENSURE_EQ(context, NumElements(bias), SizeOfDimension(filter, 0));
  }

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8 ||
      input->type == kTfLiteInt16) {
    // Populate per-channel quantization parameters, if per-channel
    // quantization.
    TF_LITE_ENSURE_EQ(context, input->quantization.type,
                      kTfLiteAffineQuantization);
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    const int per_channel_quantization_size = affine_quantization->scale->size;
    const bool is_per_channel = per_channel_quantization_size > 1;
    if (is_per_channel) {
      //  Currently only Int8/Int16 is supported for per channel quantization.
      TF_LITE_ENSURE(context,
                     input->type == kTfLiteInt8 || input->type == kTfLiteInt16);
      TF_LITE_ENSURE(context, (filter->type == kTfLiteInt8 ||
                               filter->type == kTfLiteInt4));
      TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                        per_channel_quantization_size);
      TF_LITE_ENSURE_EQ(
          context, per_channel_quantization_size,
          filter->dims->data[affine_quantization->quantized_dimension]);
      // Populate multiplier and shift using affine quantization.
      const float input_scale = input->params.scale;
      const float output_scale = output->params.scale;
      const float* filter_scales = affine_quantization->scale->data;
      data->per_channel_output_multiplier.resize(per_channel_quantization_size);
      data->per_channel_output_shift.resize(per_channel_quantization_size);
      int32_t* per_channel_multiplier =
          data->per_channel_output_multiplier.data();
      int32_t* per_channel_shift = data->per_channel_output_shift.data();
      for (int i = 0; i < per_channel_quantization_size; ++i) {
        const float scale = filter_scales[i];
        const double filter_scale = static_cast<double>(scale);
        const double effective_output_scale = static_cast<double>(input_scale) *
                                              filter_scale /
                                              static_cast<double>(output_scale);
        int32_t significand;
        int channel_shift;
        QuantizeMultiplier(effective_output_scale, &significand,
                           &channel_shift);
        per_channel_multiplier[i] = significand;
        per_channel_shift[i] = channel_shift;
      }
    } else {
      // Populate scalar quantization parameters otherwise.
      double real_multiplier = 0.0;
      TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
          context, input, filter, bias, output, &real_multiplier));
      int exponent;
      QuantizeMultiplier(real_multiplier, &data->output_multiplier, &exponent);
      data->output_shift = exponent;
    }

    if (input->type == kTfLiteUInt8 && output->type == kTfLiteInt16) {
      TF_LITE_ENSURE(context, filter->type == kTfLiteUInt8);
    }

    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, params->activation, output, &data->output_activation_min,
        &data->output_activation_max));
  }
  if (input->type == kTfLiteInt16 && output->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);

    // Check quantized_bias_type is either kTfLiteInt64 or kTfLiteInt32.
    if (params->quantized_bias_type != kTfLiteFloat32) {
      TF_LITE_ENSURE(context, params->quantized_bias_type == kTfLiteInt32 ||
                                  params->quantized_bias_type == kTfLiteInt64);
      TF_LITE_ENSURE(context, (bias == nullptr) ||
                                  bias->type == params->quantized_bias_type);
      data->quantized_bias_type = params->quantized_bias_type;
    }
  }

  // If we have to perform on-the-fly quantization (with quantized weights and
  // float inputs) first we need to quantize the inputs. Allocate a temporary
  // buffer to store the intermediate quantized values.
  // Additionally, we allocate a temporary buffer to store the accumulated
  // quantized values prior to multiplication by the scaling factor.
  const bool is_hybrid =
      (input->type == kTfLiteFloat32 &&
       (filter->type == kTfLiteUInt8 || filter->type == kTfLiteInt8 ||
        filter->type == kTfLiteInt4));
  const bool is_sparse = filter->sparsity != nullptr;

  
  LowPrecision::Method __method = LowPrecision::FullyConnected::GetMethodFromEnv();
  int _kernel_shape[2] = { filter->dims->data[1], num_units             },
      _input_shape[2]  = { batch_size           , filter->dims->data[1] },
      _output_shape[2] = { batch_size           , num_units             };
  
  LowPrecision::Shape kernel_shape = LowPrecision::get_shape(_kernel_shape, 2),
                      input_shape,
                      output_shape;
  
  if (batch_size > 1){
    input_shape  = LowPrecision::get_shape(_input_shape,  2);
    output_shape = LowPrecision::get_shape(_output_shape, 2);
  }
  else{
    _input_shape[0] = filter->dims->data[1];
    _output_shape[0] = num_units;
    input_shape  = LowPrecision::get_shape(_input_shape,  1);
    output_shape = LowPrecision::get_shape(_output_shape, 1);
  }
  // int __dims = input->dims->size;
  // int* __sizes = new int[__dims];
  // for (int i = 0; i < input->dims->size; i++)
  //   __sizes[i] = input->dims->data[i];

  // LowPrecision::Shape __shape = LowPrecision::get_shape(__sizes, __dims);

  // int __filter_dims = 2;
  // int __filter_sizes[2] = {filter->dims->data[0], filter->dims->data[1]};
  // LowPrecision::Shape __filter_shape = LowPrecision::get_shape(__filter_sizes, __filter_dims);
  // __filter_shape = __filter_shape.T();

  // int __output_dims = output->dims->size;
  // int* __output_sizes = new int[__dims];
  // for (int i = 0; i < output->dims->size; i++)
  //   __output_sizes[i] = output->dims->data[i];
  // LowPrecision::Shape __output_shape = LowPrecision::get_shape(__output_sizes, __output_dims);

  bool should_apply_low_precision = LowPrecision::FullyConnected::IsAppliable(
    __method, input_shape, kernel_shape,
    LowPrecision::FullyConnected::GetDataType(input->type),
    LowPrecision::FullyConnected::GetDataType(filter->type),
    LowPrecision::FullyConnected::GetDataType(output->type),
    input_shape.flatsize >= 2 * 2048
  );
  // std::cerr << "Method is 0x" << std::hex << ((int)__method) << std::dec << " and isApplicable is " << ((int)should_apply_low_precision) << std::endl;
  bool includes_low_precision_activation = LowPrecision::FullyConnected::IncludesActivationCompression(__method);
  if (LowPrecision::FullyConnected::GetVariableFromEnv( "ForceCaching" ) == "TRUE")
    CpuBackendContext::GetFromContext(context)->SetUseCaching(true);

  LowPrecision::ShapeList kernel_scratchpads_shape_list,
                          input_scratchpads_shape_list,
                          output_scratchpads_shape_list;
  if (should_apply_low_precision) {
    kernel_scratchpads_shape_list = LowPrecision::GetFilterShapeListForMethod(__method, kernel_shape);
    input_scratchpads_shape_list  = LowPrecision::GetInputShapeListForMethod (__method, input_shape);
    output_scratchpads_shape_list = LowPrecision::GetOutputShapeListForMethod(__method, input_shape, kernel_shape, output_shape);
  }
  int num_kernel_scratchpads = kernel_scratchpads_shape_list.size(),
      num_input_scratchpads  = input_scratchpads_shape_list.size(),
      num_output_scratchpads = output_scratchpads_shape_list.size();

  int qk_id = 0;
  int ip_id = 1;
  // int ca_id = 2;
  int num_tmps = 5;

  if (should_apply_low_precision){
    num_tmps += num_kernel_scratchpads; //(num_kernel_scratchpads > 1)?(2):(1);
    num_tmps += num_input_scratchpads ; //(num_input_scratchpads  > 0)?(1):(0);
    num_tmps += num_output_scratchpads; //(num_output_scratchpads > 0)?(1):(0);
  }

  if (is_hybrid) {
    qk_id = 5;
    ip_id = 6;
    // ca_id = 7;
    // Use optimized implementation for 4bit
    if (filter->type == kTfLiteInt4 && kernel_type == kGenericOptimized &&
        IsConstantTensor(filter) && batch_size &&
        ((input_size / batch_size) % 2 == 0) &&
        num_units >= optimized_4bit::FilterWidth &&
        (input_size / batch_size) >= optimized_4bit::FilterDepth) {
      const int cols = input_size / batch_size;
      if (!data->op_data_4bit) {
        data->op_data_4bit = std::make_unique<optimized_4bit::OpData4Bit>();
      }
      if (data->op_data_4bit->batch_size == batch_size) {
        return kTfLiteOk;
      }
      data->op_data_4bit->batch_size = batch_size;
      for (int packed_rows = optimized_4bit::GetMaxSupportedRows();
           packed_rows > 0; packed_rows /= 2) {
        if (batch_size >= packed_rows) {
          data->op_data_4bit->rows_right = packed_rows;
          break;
        }
      }
      return PrepareImpl4Bit(context, node, optimized_4bit::FilterWidth,
                             data->op_data_4bit->rows_right,
                             optimized_4bit::FilterDepth, batch_size, cols,
                             num_units);
    }
    TfLiteIntArrayFree(node->temporaries);
    data->compute_row_sums = true;
    if (is_sparse) {
      node->temporaries = TfLiteIntArrayCreate(num_tmps + 1);
    } else {
      node->temporaries = TfLiteIntArrayCreate(num_tmps);
    }
    node->temporaries->data[0] = data->scratch_tensor_index;

    TfLiteTensor* input_quantized;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/0,
                                                &input_quantized));
    input_quantized->type = kTfLiteInt8;
    input_quantized->allocation_type = kTfLiteArenaRw;

    TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                     input_quantized_size));

    node->temporaries->data[1] = data->scratch_tensor_index + 1;
    TfLiteTensor* scaling_factors;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/1,
                                                &scaling_factors));
    scaling_factors->type = kTfLiteFloat32;
    scaling_factors->allocation_type = kTfLiteArenaRw;

    int scaling_dims[1] = {batch_size};
    if (!TfLiteIntArrayEqualsArray(scaling_factors->dims, 1, scaling_dims)) {
      TfLiteIntArray* scaling_factors_size = TfLiteIntArrayCreate(1);
      scaling_factors_size->data[0] = batch_size;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scaling_factors,
                                                       scaling_factors_size));
    }

    node->temporaries->data[2] = data->scratch_tensor_index + 2;
    TfLiteTensor* accum_scratch;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/2, &accum_scratch));
    accum_scratch->type = kTfLiteInt32;
    accum_scratch->allocation_type = kTfLiteArenaRw;
    int accum_scratch_dims[2] = {num_units, batch_size};
    if (!TfLiteIntArrayEqualsArray(accum_scratch->dims, 2,
                                   accum_scratch_dims)) {
      TfLiteIntArray* accum_size = TfLiteIntArrayCreate(2);
      accum_size->data[0] = num_units;
      accum_size->data[1] = batch_size;
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, accum_scratch, accum_size));
    }

    node->temporaries->data[3] = data->scratch_tensor_index + 3;
    TfLiteTensor* input_offsets;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/3, &input_offsets));
    input_offsets->type = kTfLiteInt32;
    input_offsets->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqualsArray(input_offsets->dims, 1, scaling_dims)) {
      TfLiteIntArray* input_offsets_size = TfLiteIntArrayCreate(1);
      input_offsets_size->data[0] = batch_size;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_offsets,
                                                       input_offsets_size));
    }
    node->temporaries->data[4] = data->scratch_tensor_index + 4;
    TfLiteTensor* row_sums;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, /*index=*/4, &row_sums));
    row_sums->type = kTfLiteInt32;
    row_sums->allocation_type = kTfLiteArenaRwPersistent;
    int row_sums_dims[1] = {num_units};
    if (!TfLiteIntArrayEqualsArray(row_sums->dims, 1, row_sums_dims)) {
      TfLiteIntArray* row_sums_size = TfLiteIntArrayCreate(1);
      row_sums_size->data[0] = row_sums_dims[0];
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, row_sums, row_sums_size));
    }

    if (is_sparse) {
      data->ledger_initialized = false;
      node->temporaries->data[5] = data->scratch_tensor_index + 5;
      TfLiteTensor* filter_ledger =
          &context->tensors[node->temporaries->data[5]];
      auto status =
          CreateLedgerTensor(filter->sparsity, context, filter_ledger);
      if (status != kTfLiteOk) return status;
    }
  }
  
  data->low_precision_id = LowPrecision::FullyConnected::id++;
  if (should_apply_low_precision)
    std::cerr << "Applying FC Low-Precision for Kernel shape "
              << LowPrecision::get_shape_string(kernel_shape)
              << ", Input shape "
              << LowPrecision::get_shape_string(input_shape)
              << ", Output shape "
              << LowPrecision::get_shape_string(output_shape)
              << ", ID: "
              << data->low_precision_id
              << ", Method: "
              << LowPrecision::get_method_string(__method)
              << std::endl;
  else
    std::cerr << "NOT Applying FC Low-Precision for Kernel shape "
              << LowPrecision::get_shape_string(kernel_shape)
              << ", Input shape "
              << LowPrecision::get_shape_string(input_shape)
              << ", Output shape "
              << LowPrecision::get_shape_string(output_shape)
              << ", ID: "
              << data->low_precision_id
              << ", Method: "
              << LowPrecision::get_method_string(__method)
              << std::endl;

  data->low_precision_applicable = should_apply_low_precision;

  if (data->low_precision_applicable){
    data->operation_method = __method;
    LowPrecision::FullyConnected::set_default_method(__method);

    data->kernel_temps = num_kernel_scratchpads;
    data->input_temps  = num_input_scratchpads;
    data->output_temps = num_output_scratchpads;

    data->filter_temps_idx = qk_id;
    data->kernel_temps_idx = data->filter_temps_idx + num_kernel_scratchpads - 1; // (num_kernel_scratchpads >= 0)?(1):(0);
    data->input_temps_idx  = data->filter_temps_idx + num_kernel_scratchpads    ; // (num_input_scratchpads  >= 0)?(1):(0);
    data->output_temps_idx = data->input_temps_idx  + num_input_scratchpads     ; // (num_output_scratchpads >= 0)?(1):(0);

    data->timing_details = new LowPrecision::TimingDetailes();
    data->timing_details->activate(LowPrecision::FullyConnected::GetVariableFromEnv( "GEMMAPITiming_Disable" ) != "TRUE");

    bool k_need_preparing = false,
         i_need_preparing = false,
         o_need_preparing = false;
    
    std::cerr << "\tTotal Number of Scratchpads:"
              << " Input:  " << num_input_scratchpads
              << " Filter: " << num_kernel_scratchpads
              << " Output: " << num_output_scratchpads
              << std::endl;
    std::cerr.flush();

    // Allocating Filter and Required Kernel Scratchpads
    if (num_kernel_scratchpads >= 1){ // Filter Tensor
      int tensor_idx = data->filter_temps_idx;
      node->temporaries->data[tensor_idx] = data->scratch_tensor_index + tensor_idx;
      TfLiteTensor* tensor;
      TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/tensor_idx, &tensor));
      tensor->type = kTfLiteInt8;
      tensor->allocation_type = kTfLitePersistentRo;
      LowPrecision::Shape tensor_shape = kernel_scratchpads_shape_list.back();
      if (!TfLiteIntArrayEqualsArray(tensor->dims, tensor_shape.number_dims, tensor_shape.size)) {
        std::cerr << "\tAllocating Filter Shape: " << LowPrecision::get_shape_string(tensor_shape);
        std::cerr.flush();
        k_need_preparing = true;
        TfLiteIntArray* tensor_size = TfLiteIntArrayCreate(tensor_shape.number_dims);
        for (int d = 0; d < tensor_shape.number_dims; d++)
          tensor_size->data[d] = tensor_shape.size[d];
        TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, tensor, tensor_size));
        std::cerr << " DONE" << std::endl;
      }
    }
    for (int i = 1 ; i < num_kernel_scratchpads ; i++){ // Kernel Scratchpads Tensor
      int tensor_idx = data->kernel_temps_idx + i - 1;
      node->temporaries->data[tensor_idx] = data->scratch_tensor_index + tensor_idx;
      TfLiteTensor* tensor;
      TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/tensor_idx, &tensor));
      tensor->type = kTfLiteInt8;
      tensor->allocation_type = kTfLiteArenaRw;
      LowPrecision::Shape tensor_shape = kernel_scratchpads_shape_list[num_kernel_scratchpads - i - 1];
      if (!TfLiteIntArrayEqualsArray(tensor->dims, tensor_shape.number_dims, tensor_shape.size)) {
        std::cerr << "\tAllocating A Kernel Temporary Tensor With Shape: " << LowPrecision::get_shape_string(tensor_shape);
        std::cerr.flush();
        TfLiteIntArray* tensor_size = TfLiteIntArrayCreate(tensor_shape.number_dims);
        for (int d = 0; d < tensor_shape.number_dims; d++)
          tensor_size->data[d] = tensor_shape.size[d];
        TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, tensor, tensor_size));
        std::cerr << " DONE" << std::endl;
      }
    }
    
    // Creating Filter Matrix
    TfLiteTensor* filter_tensor;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/data->filter_temps_idx, &filter_tensor));
    int8_t* kernel_scratchpad_tensor = nullptr;
    if (num_kernel_scratchpads)
      kernel_scratchpad_tensor = LowPrecision::allocate<int8_t>(kernel_scratchpads_shape_list[num_kernel_scratchpads - 2].flatsize);
    data->filter_matrix = new LowPrecision::Matrix();
    data->filter_matrix->setDataAndPaddingAndScratchpadAndShape(
      GetTensorData<int8_t>(filter), 
      GetTensorData<int8_t>(filter_tensor), 
      kernel_scratchpad_tensor, 
      kernel_shape);
    if (num_kernel_scratchpads > 1)
        data->filter_matrix->setPaddingScratchpadSetting();
    data->filter_matrix->setNeedScratchpad();
    data->filter_matrix->setMemLayout(LowPrecision::MemLayout::kRowMajor);

    // Preparing Filter Matrix
    // LowPrecision::TimingDetailes* filter_preparation_timings = new LowPrecision::TimingDetailes();
    // filter_preparation_timings->activate();
    LowPrecision::Status filter_preparation_status;
    std::cerr << "\tPreparing Filter With Shape: " << LowPrecision::get_shape_string(kernel_shape);
    std::cerr.flush();
    filter_preparation_status = LowPrecision::PrepareMatrixAsFilterForMethod(*data->filter_matrix, __method, data->timing_details);
    std::cerr << " DONE" << std::endl;
    TF_LITE_ASSERT_EQ(LowPrecision::mask_out_source(LowPrecision::report_on_failure(filter_preparation_status, data->low_precision_id, "FC")), LowPrecision::Status::Success);
    if (num_kernel_scratchpads)
      LowPrecision::deallocate(kernel_scratchpad_tensor, true);

    // Allocating Required Input Scratchpads
    for (int i = 0 ; i < num_input_scratchpads ; i++){ // Input Scratchpads Tensor
      int tensor_idx = data->input_temps_idx + i;
      node->temporaries->data[tensor_idx] = data->scratch_tensor_index + tensor_idx;
      TfLiteTensor* tensor;
      TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/tensor_idx, &tensor));
      tensor->type = kTfLiteInt8;
      tensor->allocation_type = kTfLiteArenaRw;
      LowPrecision::Shape tensor_shape = input_scratchpads_shape_list[num_input_scratchpads - 1 - i];
      if (!TfLiteIntArrayEqualsArray(tensor->dims, tensor_shape.number_dims, tensor_shape.size)) {
        std::cerr << "\tAllocating An Input Temporary Tensor With Shape: " << LowPrecision::get_shape_string(tensor_shape);
        std::cerr.flush();
        TfLiteIntArray* tensor_size = TfLiteIntArrayCreate(tensor_shape.number_dims);
        for (int d = 0; d < tensor_shape.number_dims; d++)
          tensor_size->data[d] = tensor_shape.size[d];
        TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, tensor, tensor_size));
        std::cerr << " DONE" << std::endl;
      } else {
        std::cerr << "\tAlready Allocated the Input Temporary Tensor With Shape: " << LowPrecision::get_shape_string(tensor_shape) << std::endl;
        std::cerr.flush();
      }
    }
    
    // Allocating Required Output Scratchpads
    for (int i = 0 ; i < num_output_scratchpads ; i++){ // Output Scratchpads Tensor
      int tensor_idx = data->output_temps_idx + i;
      node->temporaries->data[tensor_idx] = data->scratch_tensor_index + tensor_idx;
      TfLiteTensor* tensor;
      TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/tensor_idx, &tensor));
      tensor->type = kTfLiteInt32;
      tensor->allocation_type = kTfLiteArenaRw;
      LowPrecision::Shape tensor_shape = output_scratchpads_shape_list[num_output_scratchpads - 1 - i];
      if (!TfLiteIntArrayEqualsArray(tensor->dims, tensor_shape.number_dims, tensor_shape.size)) {
        std::cerr << "\tAllocating An Output Temporary Tensor With Shape: " << LowPrecision::get_shape_string(tensor_shape);
        std::cerr.flush();
        TfLiteIntArray* tensor_size = TfLiteIntArrayCreate(tensor_shape.number_dims);
        for (int d = 0; d < tensor_shape.number_dims; d++)
          tensor_size->data[d] = tensor_shape.size[d];
        TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, tensor, tensor_size));
        std::cerr << " DONE" << std::endl;
      }
    }

    // node->temporaries->data[qk_id] = data->scratch_tensor_index + qk_id;
    // TfLiteTensor* quantized_kernels = GetTemporary(context, node, /*index=*/qk_id);
    // quantized_kernels->type = kTfLiteInt8;
    // quantized_kernels->allocation_type = kTfLitePersistentRo;
    // int quantized_kernels_dims[2] = {filter->dims->data[0], filter->dims->data[1]};
    // LowPrecision::FullyConnected::TransformFilterShape(__method, quantized_kernels_dims, 2);
    // std::cout << "\tTransformed Filter Shape: (" 
    //           << quantized_kernels_dims[0] << ", " << quantized_kernels_dims[1]
    //           << ")\n";
    // if (!TfLiteIntArrayEqualsArray(quantized_kernels->dims, 2, quantized_kernels_dims)) {
    //   TfLiteIntArray* quantized_kernels_size = TfLiteIntArrayCreate(2);
    //   quantized_kernels_size->data[0] = quantized_kernels_dims[0];
    //   quantized_kernels_size->data[1] = quantized_kernels_dims[1];
    //   int* _sizes = new int[2];
    //   _sizes[0] = filter->dims->data[0];
    //   _sizes[1] = filter->dims->data[1];
    //   LowPrecision::Shape _shape = LowPrecision::get_shape(_sizes, 2);
    //   TF_LITE_ENSURE_OK(
    //       context, context->ResizeTensor(context, quantized_kernels, quantized_kernels_size));
    //   LowPrecision::Status ret;
    //   ret = LowPrecision::FullyConnected::QuantizeFilter(
    //     __method, GetTensorData<int8_t>(filter),
    //     _shape, GetTensorData<int8_t>(quantized_kernels),
    //     LowPrecision::MemLayout::kRowMajor
    //   );
    //   TF_LITE_ASSERT_EQ(ret, LowPrecision::Status::Success);
    //   if (includes_low_precision_activation || batch_size > 1){
    //     data->low_precision_compress_activation = true;
    //     node->temporaries->data[ip_id] = data->scratch_tensor_index + ip_id;
    //
    //     TfLiteTensor* input_compressed = GetTemporary(context, node, /*index=*/ip_id);
    //     input_compressed->type = filter->type;
    //     input_compressed->allocation_type = kTfLiteArenaRw;
    //     std::cout << "\tTransformed Activation Shape From: ("
    //               << __sizes[0] << ", " << __sizes[1]
    //               << ") ";
    //     LowPrecision::FullyConnected::TransformInputShape(__method, __sizes, 2);
    //     std::cout << "To: ("
    //               << __sizes[0] << ", " << __sizes[1]
    //               << ")\n";
    //
    //     TfLiteIntArray* compressed_activation_size = TfLiteIntArrayCreate(2);
    //     compressed_activation_size->data[0] = __sizes[0];
    //     compressed_activation_size->data[1] = __sizes[1];
    //     TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_compressed,
    //                                                     compressed_activation_size));
    //   }
    // }
  }

  // Resize output.
  return UpdateOutputSize(context, params, input, output, batch_size, num_units,
                          filter->dims->data[1]);
}

template <KernelType kernel_type>
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  if (data->scratch_tensor_index == -1) {
    context->AddTensors(context, /*tensors_to_add=*/20,
                        &data->scratch_tensor_index);
  }
  // Check for supported activation types.
  auto* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
  const TfLiteTensor* filter;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kWeightsTensor, &filter));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const bool is_quantized =
      ((filter->type == kTfLiteUInt8) || (filter->type == kTfLiteInt8) ||
       (filter->type == kTfLiteInt4));
  const bool is_hybrid = is_quantized && (input->type == kTfLiteFloat32);
  const bool is_pie = kernel_type == kLegacyPie;

  // Pie and hybrid path supports all kinds of fused activations, otherwise only
  // clipping activations are supported.
  if (!is_pie && !is_hybrid) {
    TF_LITE_ENSURE(context, params->activation == kTfLiteActNone ||
                                params->activation == kTfLiteActRelu ||
                                params->activation == kTfLiteActReluN1To1 ||
                                params->activation == kTfLiteActRelu6);
  }
  if (filter->type == kTfLiteInt4) {
    TF_LITE_ENSURE_MSG(
        context,
        kTfLiteOk == VerifyQuantizationZeroPoint(filter, /*expected_value=*/0),
        "Unsupported filter quantization zero-point value.");
  }
  return PrepareImpl(context, node, kernel_type);
}

TfLiteStatus EvalPie(TfLiteContext* context, TfLiteNode* node,
                     TfLiteFullyConnectedParams* params, OpData* data,
                     const TfLiteTensor* input, const TfLiteTensor* filter,
                     const TfLiteTensor* bias, TfLiteTensor* output) {
  int total_input_size = 1;
  for (int i = 0; i < input->dims->size; i++) {
    total_input_size *= input->dims->data[i];
  }

  int input_size = filter->dims->data[1];
  const int batch_size = total_input_size / filter->dims->data[1];
  const int num_units = filter->dims->data[0];

  // Output = bias if bias tensor exists.
  if (bias) {
    tensor_utils::VectorBatchVectorAssign(GetTensorData<float>(bias), num_units,
                                          batch_size,
                                          GetTensorData<float>(output));
  } else {
    std::fill_n(GetTensorData<float>(output), batch_size * num_units, 0.0f);
  }

  // Compute output += weight * input
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      GetTensorData<float>(filter), num_units, input_size,
      GetTensorData<float>(input), batch_size, GetTensorData<float>(output));

  // Apply activation function
  tensor_utils::ApplyActivationToVector(
      GetTensorData<float>(output), batch_size * num_units, params->activation,
      GetTensorData<float>(output));

  return kTfLiteOk;
}

TfLiteStatus EvalHybridDense(
    TfLiteContext* context, TfLiteNode* node,
    TfLiteFullyConnectedParams* params, OpData* data, const TfLiteTensor* input,
    const TfLiteTensor* filter, const TfLiteTensor* bias,
    TfLiteTensor* input_quantized, TfLiteTensor* scaling_factors,
    TfLiteTensor* accum_scratch, TfLiteTensor* row_sums,
    TfLiteTensor* input_offsets, TfLiteTensor* output) {
  int total_input_size = 1;
  for (int i = 0; i < input->dims->size; i++) {
    total_input_size *= input->dims->data[i];
  }

  const int input_size = filter->dims->data[1];
  const int batch_size = total_input_size / filter->dims->data[1];
  const int num_units = filter->dims->data[0];

  // Output = bias if bias tensor exists.
  if (bias) {
    tensor_utils::VectorBatchVectorAssign(GetTensorData<float>(bias), num_units,
                                          batch_size,
                                          GetTensorData<float>(output));
  } else {
    std::fill_n(GetTensorData<float>(output), batch_size * num_units, 0.0f);
  }

  // Save matrix multiplication computation for all zero input.
  if (tensor_utils::IsZeroVector(GetTensorData<float>(input),
                                 total_input_size)) {
    tensor_utils::ApplyActivationToVector(
        GetTensorData<float>(output), batch_size * num_units,
        params->activation, GetTensorData<float>(output));
    return kTfLiteOk;
  }

  // Quantize input from float to uint8 + quantization params (scaling factor).
  float* scaling_factors_ptr = GetTensorData<float>(scaling_factors);
  int32_t* input_offset_ptr = nullptr;
  int32_t* row_sums_ptr = nullptr;
  if (params->asymmetric_quantize_inputs) {
    input_offset_ptr = GetTensorData<int32_t>(input_offsets);
    row_sums_ptr = GetTensorData<int32_t>(row_sums);
  }
  int8_t* quant_data = GetTensorData<int8_t>(input_quantized);
  const int8_t* filter_data = nullptr;
  std::unique_ptr<int8_t[]> unpacked_filter_data = nullptr;
  // Unoptimized 4-bit implementation. Ideally use EvalHybridDenseInt4 instead.
  if (filter->type == kTfLiteInt4) {
    const size_t bytes_unpacked = filter->bytes * 2;
    unpacked_filter_data = std::make_unique<int8_t[]>(bytes_unpacked);
    tflite::tensor_utils::UnpackDenseInt4IntoInt8(
        GetTensorData<int8_t>(filter), GetTensorShape(filter).FlatSize(),
        unpacked_filter_data.get());
    filter_data = unpacked_filter_data.get();
  } else {
    filter_data = GetTensorData<int8_t>(filter);
  }
  const float* input_ptr = GetTensorData<float>(input);
  tensor_utils::BatchQuantizeFloats(
      input_ptr, batch_size, input_size, quant_data, scaling_factors_ptr,
      input_offset_ptr, params->asymmetric_quantize_inputs);

  float* per_channel_scale_ptr = nullptr;
  if (VerifyPerChannelQuantization(context, filter) == kTfLiteOk) {
    //  Per channel quantization.
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    TF_LITE_ENSURE_EQ(
        context, affine_quantization->scale->size,
        filter->dims->data[affine_quantization->quantized_dimension]);
    per_channel_scale_ptr = affine_quantization->scale->data;
  } else {
    // Per tensor quantization.
    for (int b = 0; b < batch_size; ++b) {
      // Incorporate scaling of the filter
      scaling_factors_ptr[b] *= filter->params.scale;
    }
  }
  
  int32_t* scratch = GetTensorData<int32_t>(accum_scratch);

  if (data->low_precision_applicable){
    LowPrecision::Method method = LowPrecision::FullyConnected::get_default_method();
    int _kernel_shape[2] = { input_size, num_units  },
        _input_shape[2]  = { batch_size, input_size },
        _output_shape[2] = { batch_size, num_units  };
    
    LowPrecision::Shape kernel_shape = LowPrecision::get_shape(_kernel_shape, 2),
                        input_shape,
                        output_shape;
    
    if (batch_size > 1){
      input_shape  = LowPrecision::get_shape(_input_shape,  2);
      output_shape = LowPrecision::get_shape(_output_shape, 2);
    }
    else{
      _input_shape[0] = input_size;
      _output_shape[0] = num_units;
      input_shape  = LowPrecision::get_shape(_input_shape,  1);
      output_shape = LowPrecision::get_shape(_output_shape, 1);
    }
    
    // Creating Timing Profilers
    // LowPrecision::TimingDetailes* gemm_timing_profiler              = new LowPrecision::TimingDetailes();
    // LowPrecision::TimingDetailes* input_preprocess_timing_profiler  = new LowPrecision::TimingDetailes();
    // LowPrecision::TimingDetailes* output_preprocess_timing_profiler = new LowPrecision::TimingDetailes();
    // gemm_timing_profiler              ->activate();
    // input_preprocess_timing_profiler  ->activate();
    // output_preprocess_timing_profiler ->activate();


    // Getting Kernel Temporary Tensors
    TfLiteTensor* filter_tensor = GetTemporary(context, node, /*index=*/data->filter_temps_idx);

    // Creating Filter Matrix
    // LowPrecision::Matrix filter_matrix;
    // filter_matrix.setDataAndPaddingAndScratchpadAndShape(
    //   GetTensorData<int8_t>(filter), 
    //   GetTensorData<int8_t>(filter_tensor), 
    //   nullptr,
    //   kernel_shape);
    // filter_matrix.setNeedScratchpad();
    // filter_matrix.setScratchpadValid();
    // filter_matrix.setMemLayout(LowPrecision::MemLayout::kRowMajor);

    // Getting Input Temporary Tensors
    std::vector<TfLiteTensor*> input_scratchpads(data->input_temps, nullptr);
    for (size_t i = 0; i < data->input_temps; i++){
      TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/data->input_temps_idx + i, &input_scratchpads[i]));
    }

    // Creating Input Matrix
    LowPrecision::Matrix input_matrix;
    input_matrix.setData(quant_data);
    input_matrix.setShape(input_shape);
    if (data->input_temps >= 1){
      input_matrix.setScratchpad(GetTensorData<int8_t>(input_scratchpads[0]));
      input_matrix.setNeedScratchpad();
    }
    if (data->input_temps >= 2){
      input_matrix.setPaddingScratchpad(GetTensorData<int8_t>(input_scratchpads[1]));
      input_matrix.setPaddingScratchpadSetting();
    }
    input_matrix.setMemLayout(LowPrecision::MemLayout::kRowMajor);

    // Preparing Input Matrix
    LowPrecision::Status input_preparation_status;
    input_preparation_status = LowPrecision::PrepareMatrixAsInputForMethod(input_matrix, method, data->timing_details);
    TF_LITE_ASSERT_EQ(LowPrecision::mask_out_source(LowPrecision::report_on_failure(input_preparation_status, data->low_precision_id, "FC")), LowPrecision::Status::Success);
    
    // Getting Output Temporary Tensors
    TfLiteTensor** output_scratchpads = nullptr;
    output_scratchpads = new TfLiteTensor*[data->output_temps];
    for (size_t i = 0; i < data->output_temps; i++)
      output_scratchpads[i] = GetTemporary(context, node, /*index=*/data->output_temps_idx + i);

    // Creating Output Matrix
    LowPrecision::Matrix output_matrix;
    output_matrix.setData(scratch);
    output_matrix.setShape(output_shape);
    if (data->output_temps >= 1){
      if (LowPrecision::FullyConnected::OutputPreProcess(method) & LowPrecision::PreprocessType::Packing){
        output_matrix.setScratchpad(GetTensorData<int32_t>(output_scratchpads[0]));
        output_matrix.setNeedScratchpad();
      } else {
        output_matrix.setPaddingScratchpad(GetTensorData<int32_t>(output_scratchpads[0]));
        output_matrix.setPaddingScratchpadSetting();
      }
    }
    if (data->output_temps >= 2){
      output_matrix.setPaddingScratchpad(GetTensorData<int32_t>(output_scratchpads[1]));
      output_matrix.setPaddingScratchpadSetting();
    }
    output_matrix.setMemLayout(LowPrecision::MemLayout::kRowMajor);

    // Preparing Output Matrix
    LowPrecision::Status output_preparation_status;
    output_preparation_status = LowPrecision::PrepareMatrixAsOutputForMethod(output_matrix, method, data->timing_details);
    TF_LITE_ASSERT_EQ(LowPrecision::mask_out_source(LowPrecision::report_on_failure(output_preparation_status, data->low_precision_id, "FC")), LowPrecision::Status::Success);

    LowPrecision::Status gemm_status;
    gemm_status = LowPrecision::GEMM(input_matrix, *data->filter_matrix, output_matrix, method, data->timing_details);
    TF_LITE_ASSERT_EQ(LowPrecision::mask_out_source(LowPrecision::report_on_failure(gemm_status, data->low_precision_id, "FC")), LowPrecision::Status::Success);

    if (scaling_factors_ptr != nullptr){
      LowPrecision::doScallingFactorMultiplication(
        scratch, scaling_factors_ptr, GetTensorData<float>(output),
        batch_size, num_units
      );
    }
  }
  else {
    // TfLiteTensor* filters_int4 = nullptr;
    // TfLiteTensor* packed_input = nullptr;
    // if (data->low_precision_applicable){
    //   filters_int4 = GetTemporary(context, node, /*index=*/5);
    // }
    // if (data->low_precision_compress_activation){
    //   packed_input = GetTemporary(context, node, /*index=*/6);
    // }
    bool false_bool = false;
    // Compute output += weight * quantized_input
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        filter_data, num_units, input_size, quant_data, scaling_factors_ptr,
        batch_size, GetTensorData<float>(output), per_channel_scale_ptr,
        input_offset_ptr, scratch, row_sums_ptr, &data->compute_row_sums,
        CpuBackendContext::GetFromContext(context));
  }
  // Apply activation function to floats.
  tensor_utils::ApplyActivationToVector(
      GetTensorData<float>(output), batch_size * num_units, params->activation,
      GetTensorData<float>(output));
  return kTfLiteOk;
}

void EvalSparseHybridImpl(TfLiteContext* context, TfLiteNode* node,
                          TfLiteFullyConnectedParams* params, OpData* data,
                          const TfLiteTensor* input, const TfLiteTensor* filter,
                          const TfLiteTensor* bias, int thread_start,
                          int thread_end, TfLiteTensor* input_quantized,
                          TfLiteTensor* scaling_factors,
                          TfLiteTensor* accum_scratch, TfLiteTensor* row_sums,
                          TfLiteTensor* input_offsets, TfLiteTensor* output) {
  ruy::profiler::ScopeLabel label("FullyConnected");
  ruy::profiler::ScopeLabel inner_label("Sparse Hybrid Kernel");
  const auto& input_shape = GetTensorShape(input);
  const auto& output_shape = GetTensorShape(output);
  const auto& filter_shape = GetTensorShape(filter);
  const int input_dims_count = input_shape.DimensionsCount();
  const int output_dims_count = output_shape.DimensionsCount();
  const int filter_dims_count = filter_shape.DimensionsCount();
  const int batch_size = thread_end - thread_start;
  const int input_depth = MatchingDim(filter_shape, filter_dims_count - 1,
                                      input_shape, input_dims_count - 1);
  const int output_depth = MatchingDim(filter_shape, filter_dims_count - 2,
                                       output_shape, output_dims_count - 1);
  const int per_thread_input_size = batch_size * input_depth;

  const float* per_thread_input =
      GetTensorData<float>(input) + thread_start * input_depth;
  float* per_thread_output =
      GetTensorData<float>(output) + thread_start * output_depth;

  // Output = bias if bias tensor exists.
  if (bias) {
    tensor_utils::VectorBatchVectorAssign(GetTensorData<float>(bias),
                                          output_depth, batch_size,
                                          per_thread_output);
  } else {
    std::fill_n(per_thread_output, batch_size * output_depth, 0.0f);
  }

  // Save matrix multiplication computation for all zero input.
  if (tensor_utils::IsZeroVector(per_thread_input, per_thread_input_size)) {
    tensor_utils::ApplyActivationToVector(
        per_thread_output, batch_size * output_depth, params->activation,
        per_thread_output);
    return;
  }

  // Quantize input from float to uint8 + quantization params (scaling factor).
  float* scaling_factors_ptr =
      GetTensorData<float>(scaling_factors) + thread_start;
  int32_t* input_offset_ptr = nullptr;
  int32_t* row_sums_ptr = nullptr;
  if (params->asymmetric_quantize_inputs) {
    input_offset_ptr = GetTensorData<int32_t>(input_offsets) + thread_start;
    row_sums_ptr = GetTensorData<int32_t>(row_sums);
  }
  int8_t* quant_data =
      GetTensorData<int8_t>(input_quantized) + thread_start * input_depth;
  tensor_utils::BatchQuantizeFloats(per_thread_input, batch_size, input_depth,
                                    quant_data, scaling_factors_ptr,
                                    input_offset_ptr,
                                    params->asymmetric_quantize_inputs);
  float* per_channel_scale_ptr = nullptr;
  if (VerifyPerChannelQuantization(context, filter) == kTfLiteOk) {
    //  Per channel quantization.
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    per_channel_scale_ptr = affine_quantization->scale->data;
  } else {
    // Per tensor quantization.
    for (int b = 0; b < batch_size; ++b) {
      // Incorporate scaling of the filter.
      scaling_factors_ptr[b] *= filter->params.scale;
    }
  }

  if (params->asymmetric_quantize_inputs) {
    float* per_thread_output_ptr = per_thread_output;
    for (int b = 0; b < batch_size; ++b) {
      const float scaled_zp = scaling_factors_ptr[b] * input_offset_ptr[b];
      for (int row = 0; row < output_depth; ++row) {
        float scale = scaled_zp;
        if (per_channel_scale_ptr) {
          scale *= per_channel_scale_ptr[row];
        }
        *per_thread_output_ptr++ -= scale * row_sums_ptr[row];
      }
    }
  }

  // Compute output += weight * quantized_input
  TfLiteTensor* filter_ledger = &context->tensors[node->temporaries->data[5]];
  tensor_utils::SparseMatrixBatchVectorMultiplyAccumulate(
      GetTensorData<int8_t>(filter), GetTensorData<uint8_t>(filter_ledger),
      output_depth, input_depth, quant_data, scaling_factors_ptr, batch_size,
      per_thread_output, per_channel_scale_ptr);

  // Apply activation function to floats.
  tensor_utils::ApplyActivationToVector(per_thread_output,
                                        batch_size * output_depth,
                                        params->activation, per_thread_output);
}

struct SparseHybridFullyConnectedTask : cpu_backend_threadpool::Task {
  SparseHybridFullyConnectedTask(
      TfLiteContext* context, TfLiteNode* node,
      TfLiteFullyConnectedParams* params, OpData* data,
      const TfLiteTensor* input, const TfLiteTensor* filter,
      const TfLiteTensor* bias, const int thread_start, const int thread_end,
      TfLiteTensor* input_quantized, TfLiteTensor* scaling_factors,
      TfLiteTensor* accum_scratch, TfLiteTensor* row_sums,
      TfLiteTensor* input_offsets, TfLiteTensor* output)
      : context(context),
        node(node),
        params(params),
        data(data),
        input(input),
        filter(filter),
        bias(bias),
        thread_start(thread_start),
        thread_end(thread_end),
        input_quantized(input_quantized),
        scaling_factors(scaling_factors),
        accum_scratch(accum_scratch),
        row_sums(row_sums),
        input_offsets(input_offsets),
        output(output) {}

  void Run() override {
    EvalSparseHybridImpl(context, node, params, data, input, filter, bias,
                         thread_start, thread_end, input_quantized,
                         scaling_factors, accum_scratch, row_sums,
                         input_offsets, output);
  }

 private:
  TfLiteContext* context;
  TfLiteNode* node;
  TfLiteFullyConnectedParams* params;
  OpData* data;
  const TfLiteTensor* input;
  const TfLiteTensor* filter;
  const TfLiteTensor* bias;
  const int thread_start;
  const int thread_end;
  TfLiteTensor* input_quantized;
  TfLiteTensor* scaling_factors;
  TfLiteTensor* accum_scratch;
  TfLiteTensor* row_sums;
  TfLiteTensor* input_offsets;
  TfLiteTensor* output;
};

inline int8_t SignExtendInt4(int8_t value) { return (value ^ 0x8) - 8; }

TfLiteStatus EvalBlockwise4Bit(
    TfLiteContext* context, TfLiteNode* node,
    TfLiteFullyConnectedParams* params, OpData* data, const TfLiteTensor* input,
    const TfLiteTensor* filter, const TfLiteTensor* bias,
    TfLiteTensor* input_quantized, TfLiteTensor* scaling_factors,
    TfLiteTensor* accum_scratch, TfLiteTensor* input_offsets,
    TfLiteTensor* output) {
  const auto quantization_params =
      static_cast<const TfLiteBlockwiseQuantization*>(
          filter->quantization.params);

  const size_t blocksize = quantization_params->blocksize;
  const size_t input_channels = filter->dims->data[1];
  const size_t output_channels = filter->dims->data[0];
  const size_t batch_size = NumElements(input) / input_channels;
  const size_t num_blocks = input_channels / blocksize;
  const TfLiteTensor& scale = context->tensors[quantization_params->scale];
  int num_scales = NumElements(&scale);
  std::vector<float> dequantized_scale(num_scales, 0);
  const Eigen::half* half_data = reinterpret_cast<const Eigen::half*>(
      GetTensorData<TfLiteFloat16>(&scale));
  reference_ops::Dequantize(GetTensorShape(&scale), half_data,
                            GetTensorShape(&scale), dequantized_scale.data());
  float* output_ptr = GetTensorData<float>(output);
  memset(output_ptr, 0, NumElements(output) * sizeof(float));
  std::vector<int8_t> quant_data(NumElements(input));
  std::vector<float> input_scales(batch_size);
  std::vector<int32_t> input_zero_points(batch_size);

  const float* input_ptr = GetTensorData<float>(input);
  tensor_utils::BatchQuantizeFloats(input_ptr, batch_size, input_channels,
                                    quant_data.data(), input_scales.data(),
                                    input_zero_points.data(),
                                    /*do_asymmetric=*/true);

  const float* bias_data = nullptr;
  if (bias) {
    bias_data = GetTensorData<float>(bias);
  }
  const size_t k2 = (input_channels + 1) & 0xFFFFFFFFFFFFFFFE;
  const uint8_t* kernel = GetTensorData<uint8_t>(filter);
  for (size_t mi = 0; mi < batch_size; mi++) {
    for (size_t ni = 0; ni < output_channels; ni++) {
      float kfsum = 0.0;
      for (size_t bi = 0; bi < num_blocks; bi++) {
        int32_t ksum = 0;
        int32_t c_ref_acc = 0;
        for (size_t ki = 0; ki < blocksize; ki++) {
          const size_t k_index = bi * blocksize + ki;
          const size_t nb_index = (ni * k2 + k_index) / 2;
          const int8_t k_value = int8_t(
              (k_index % 2 == 0) ? (kernel[nb_index] & static_cast<int8_t>(0xF))
                                 : (kernel[nb_index] >> 4));
          const int32_t kernel_value = SignExtendInt4(k_value);
          ksum += kernel_value;
          c_ref_acc +=
              static_cast<int32_t>(quant_data[mi * input_channels + k_index]) *
              static_cast<float>(kernel_value);
        }
        size_t scale_index = ni * num_blocks + bi;
        float scale = dequantized_scale[scale_index];
        output_ptr[mi * output_channels + ni] += c_ref_acc * scale;
        kfsum += scale * ksum;
      }
      output_ptr[mi * output_channels + ni] -= (input_zero_points[mi] * kfsum);
      output_ptr[mi * output_channels + ni] *= input_scales[mi];
      if (bias_data != nullptr) {
        output_ptr[mi * output_channels + ni] += bias_data[ni];
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus EvalHybridDense4Bit(
    TfLiteContext* context, TfLiteNode* node,
    TfLiteFullyConnectedParams* params, OpData* data, const TfLiteTensor* input,
    const TfLiteTensor* filter, const TfLiteTensor* bias,
    TfLiteTensor* input_quantized, TfLiteTensor* scaling_factors,
    TfLiteTensor* accum_scratch, TfLiteTensor* input_offsets,
    TfLiteTensor* output) {
  float* scaling_factors_ptr = GetTensorData<float>(scaling_factors);
  int8_t* quant_data = GetTensorData<int8_t>(input_quantized);
  int32_t* input_offset_ptr = GetTensorData<int32_t>(input_offsets);
  const int batch_size = data->op_data_4bit->batch_size;
  const int output_depth = filter->dims->data[0];
  const int cols = filter->dims->data[1];
  const int rhs_width = data->op_data_4bit->rows_right;
  const int depth = optimized_4bit::FilterDepth;
  const int lhs_width = optimized_4bit::FilterWidth;
  const int lhs_layout_rows =
      (output_depth + (lhs_width - 1)) & ~(lhs_width - 1);
  const int lhs_layout_cols = (cols + (depth - 1)) & ~(depth - 1);
  const int rhs_layout_rows = (batch_size + (rhs_width - 1)) & ~(rhs_width - 1);
  const int rhs_layout_cols = lhs_layout_cols;
  const int dst_layout_rows = rhs_layout_rows;
  const int dst_layout_cols = lhs_layout_rows;
  if (data->op_data_4bit->needs_prepack) {
    const int weight_size = lhs_layout_rows * lhs_layout_cols / 2;
    const int required_size =
        optimized_4bit::kDefaultAlignmentPadding + weight_size;
    data->op_data_4bit->AllocatePackedRegion(required_size);
    const int8_t* weight_ptr = GetTensorData<int8_t>(filter);
    optimized_4bit::api::Prepack(data->op_data_4bit->prepacked_cache,
                                 weight_ptr, lhs_layout_rows, lhs_layout_cols,
                                 output_depth, cols, lhs_width, depth);
    data->op_data_4bit->needs_prepack = false;
#ifdef MADV_PAGEOUT
    // After prepacking, we will never use the weights from the model file. Mark
    // them with MADV_PAGEOUT so the kernel can reclaim the pages, decreasing
    // the resident memory size.
    //
    // This is Linux specific. There is no effect on other platforms (e.g. on
    // Windows, but possibly other POSIX platforms!). It requires a minimum
    // Kernel version of 5.4 - on older kernels the call will return with an
    // error, but we ignore it. The kernel might also ignore this hint.
    //
    // Note, due to rounding the pointer up (which is necessary due to madvise
    // requiring an address that aligns with the page size), the first partial
    // page will not be reclaimed. Madvise also rounds the end of the hinted
    // range down, so the last partial page is also unaffected. Because of this
    // behavior, on average one memory page (usually 4 kiB) per buffer holding 4
    // bit data will not be paged out.
    static const uintptr_t pagesize = sysconf(_SC_PAGESIZE);
    int8_t* up_aligned_ptr = reinterpret_cast<int8_t*>(
        ((reinterpret_cast<uintptr_t>(weight_ptr) + pagesize - 1) / pagesize) *
        pagesize);
    const auto rounding_size = up_aligned_ptr - weight_ptr;
    madvise(up_aligned_ptr, weight_size - rounding_size, MADV_PAGEOUT);
#endif
  }

  std::vector<float> filter_scales(lhs_layout_rows, filter->params.scale);
  auto* filter_params =
      reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
  if (filter_params && filter_params->scale && filter_params->scale->size > 0) {
    if (filter_params->scale->size == 1) {
      std::fill(filter_scales.begin(), filter_scales.end(),
                filter_params->scale->data[0]);
    } else {
      for (int i = 0; i < filter_params->scale->size; i++) {
        filter_scales[i] = filter_params->scale->data[i];
      }
    }
  }
  optimized_4bit::api::BatchQuantizeFloats4Bit(
      GetTensorData<float>(input), batch_size, cols, quant_data,
      scaling_factors_ptr, rhs_width, depth, input_offset_ptr);
  const float* bias_ptr =
      bias != nullptr ? GetTensorData<float>(bias) : nullptr;
  optimized_4bit::api::AssignBiasAndComputeOffsets(
      input_offset_ptr, scaling_factors_ptr, filter_scales.data(), bias_ptr,
      GetTensorData<float>(output), output_depth, batch_size);
  const uint8_t* lhs = data->op_data_4bit->prepacked_cache;
  int32_t* dst = GetTensorData<int32_t>(accum_scratch);
  optimized_4bit::api::RunAndUnpack(
      data->op_data_4bit->rows_right, lhs, quant_data, dst, output_depth,
      batch_size, lhs_layout_rows, lhs_layout_cols, rhs_layout_rows,
      rhs_layout_cols, dst_layout_rows, dst_layout_cols,
      GetTensorData<float>(output), scaling_factors_ptr, filter_scales.data());
  tensor_utils::ApplyActivationToVector(
      GetTensorData<float>(output), batch_size * output_depth,
      params->activation, GetTensorData<float>(output));
  return kTfLiteOk;
}

TfLiteStatus EvalHybrid(TfLiteContext* context, TfLiteNode* node,
                        TfLiteFullyConnectedParams* params, OpData* data,
                        const TfLiteTensor* input, const TfLiteTensor* filter,
                        const TfLiteTensor* bias, TfLiteTensor* input_quantized,
                        TfLiteTensor* scaling_factors,
                        TfLiteTensor* accum_scratch, TfLiteTensor* row_sums,
                        TfLiteTensor* input_offsets, TfLiteTensor* output) {
  const auto& output_shape = GetTensorShape(output);
  CpuBackendContext* cpu_backend_context =
      CpuBackendContext::GetFromContext(context);
  const bool is_dense = filter->sparsity == nullptr;
  if (is_dense) {
    return EvalHybridDense(context, node, params, data, input, filter, bias,
                           input_quantized, scaling_factors, accum_scratch,
                           row_sums, input_offsets, output);
  }

  TfLiteTensor* filter_ledger = &context->tensors[node->temporaries->data[5]];
  if (!data->ledger_initialized) {
    PopulateLedgerData(filter->sparsity, context,
                       GetTensorData<uint8_t>(filter_ledger));
    data->ledger_initialized = true;
  }

  // The multi-threaded kernel slices the workload along the batch dimension. If
  // there's not enough batches of data, the number of threads used is equal to
  // the batch size.
  // TODO(b/173442777): If needed, we can improve this later with slicing along
  // the row dimension of the weight.
  const int max_threads = cpu_backend_context->max_num_threads();
  const int batches =
      FlatSizeSkipDim(output_shape, output_shape.DimensionsCount() - 1);
  const int thread_count = std::max(1, std::min(batches, max_threads));
  if (params->asymmetric_quantize_inputs && data->compute_row_sums) {
    // Precompute row sums.
    static const int kBlockSize = 16;
    const uint8_t* ledger_ptr = GetTensorData<uint8_t>(filter_ledger);
    const int8_t* row_ptr = GetTensorData<int8_t>(filter);
    const int output_depth = filter->dims->data[0];
    int32_t* row_sums_ptr = GetTensorData<int32_t>(row_sums);
    for (int row = 0; row < output_depth; ++row) {
      int32_t row_sum = 0;
      int num_nonzero_blocks = *ledger_ptr++;
      for (int i = 0; i < num_nonzero_blocks; ++i, ++ledger_ptr) {
        for (int c = 0; c < kBlockSize; c++) {
          row_sum += (*row_ptr++);
        }
      }
      row_sums_ptr[row] = row_sum;
    }
    data->compute_row_sums = false;
  }
  std::vector<SparseHybridFullyConnectedTask> tasks;
  tasks.reserve(thread_count);
  int thread_start = 0;
  for (int i = 0; i < thread_count; ++i) {
    // This makes sure the workload is relatively balanced when batches is not
    // a multiple of thread_count. The first mod(batches, thread_count) tasks
    // need to process one more batch than the rest.
    int thread_end = thread_start + batches / thread_count;
    if (i < batches % thread_count) thread_end++;

    tasks.emplace_back(context, node, params, data, input, filter, bias,
                       thread_start, thread_end, input_quantized,
                       scaling_factors, accum_scratch, row_sums, input_offsets,
                       output);
    thread_start = thread_end;
  }
  cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),
                                  cpu_backend_context);
  return kTfLiteOk;
}

namespace {
template <KernelType kernel_type>
void FullyConnectedInt8(const OpData* data, const TfLiteTensor* input,
                        const TfLiteTensor* filter, const int8_t* filter_data,
                        const TfLiteTensor* bias,
                        TfLiteTensor* output, bool* low_precision_applicable,
                        TfLiteTensor* low_precision_filter, TfLiteTensor* activation,
                        CpuBackendContext* cpu_backend_context) {
  FullyConnectedParams op_params;
  op_params.input_offset = -input->params.zero_point;
  op_params.weights_offset = -filter->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  op_params.lhs_cacheable = IsConstantTensor(filter);
  op_params.rhs_cacheable = IsConstantTensor(input);
  if (*low_precision_applicable){
    optimized_integer_ops::FullyConnected(
        op_params,
        GetTensorShape(input), GetTensorData<int8_t>(input),
        GetTensorData<int8_t>(activation),
        GetTensorShape(filter), GetTensorData<int8_t>(low_precision_filter),
        GetTensorShape(output), GetTensorData<int8_t>(output));
  }
  else if (kernel_type == kReference) {
    reference_integer_ops::FullyConnected(
        op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
        GetTensorShape(filter), filter_data, GetTensorShape(bias),
        GetTensorData<int32_t>(bias), GetTensorShape(output),
        input->params.scale, output->params.scale, filter->params.scale,
        GetTensorData<int8_t>(output));
  } else {
    optimized_integer_ops::FullyConnected(
        op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
        GetTensorShape(filter), filter_data, GetTensorShape(bias),
        GetTensorData<int32_t>(bias), GetTensorShape(output),
        GetTensorData<int8_t>(output), cpu_backend_context);
  }
}

template <KernelType kernel_type>
void FullyConnectedInt16(const OpData* data, const TfLiteTensor* input,
                         const TfLiteTensor* filter, const int8_t* filter_data,
                         const TfLiteTensor* bias, TfLiteTensor* output) {
  FullyConnectedParams op_params;
  op_params.input_offset = -input->params.zero_point;
  op_params.weights_offset = -filter->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

  if (data->quantized_bias_type == kTfLiteInt32) {
    reference_integer_ops::FullyConnected(
        op_params, GetTensorShape(input), GetTensorData<int16_t>(input),
        GetTensorShape(filter), filter_data, GetTensorShape(bias),
        GetTensorData<int32_t>(bias), GetTensorShape(output),
        input->params.scale, output->params.scale, filter->params.scale,
        GetTensorData<int16_t>(output));
  } else {
    reference_integer_ops::FullyConnected(
        op_params, GetTensorShape(input), GetTensorData<int16_t>(input),
        GetTensorShape(filter), filter_data, GetTensorShape(bias),
        GetTensorData<int64_t>(bias), GetTensorShape(output),
        input->params.scale, output->params.scale, filter->params.scale,
        GetTensorData<int16_t>(output));
  }
}

template <KernelType kernel_type>
void FullyConnectedPerChannelInt8(const OpData* data, const TfLiteTensor* input,
                                  const TfLiteTensor* filter,
                                  const int8_t* filter_data,
                                  const TfLiteTensor* bias,
                                  TfLiteTensor* output,
                                  CpuBackendContext* cpu_backend_context) {
  // FullyConnectedPerChannel ops spec is that weights are symmetric.
  // op_params.weights_offset is not set (filter.params.zero_point is not used),
  // since it will be always assumed to be 0.
  FullyConnectedParams op_params;
  op_params.input_offset = -input->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  op_params.lhs_cacheable = IsConstantTensor(filter);
  op_params.rhs_cacheable = IsConstantTensor(input);

  if (kernel_type == kReference) {
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    const float* filter_scales = affine_quantization->scale->data;
    reference_integer_ops::FullyConnectedPerChannel(
        op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
        GetTensorShape(filter), filter_data, GetTensorShape(bias),
        GetTensorData<int32_t>(bias), GetTensorShape(output),
        input->params.scale, output->params.scale, filter_scales,
        GetTensorData<int8_t>(output));
  } else {
    optimized_integer_ops::FullyConnectedPerChannel(
        op_params, data->per_channel_output_multiplier.data(),
        data->per_channel_output_shift.data(), GetTensorShape(input),
        GetTensorData<int8_t>(input), GetTensorShape(filter), filter_data,
        GetTensorShape(bias), GetTensorData<int32_t>(bias),
        GetTensorShape(output), GetTensorData<int8_t>(output),
        cpu_backend_context);
  }
}

template <KernelType kernel_type>
void FullyConnectedPerChannelInt16(
    const OpData* data, const TfLiteTensor* input, const TfLiteTensor* filter,
    const int8_t* filter_data, const TfLiteTensor* bias, TfLiteTensor* output) {
  // FullyConnectedPerChannel ops spec is that weights are symmetric.
  // op_params.weights_offset is not set (filter.params.zero_point is not used),
  // since it will be always assumed to be 0.
  FullyConnectedParams op_params;
  op_params.input_offset = -input->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
  const float* filter_scales = affine_quantization->scale->data;

  if (data->quantized_bias_type == kTfLiteInt32) {
    reference_integer_ops::FullyConnectedPerChannel(
        op_params, GetTensorShape(input), GetTensorData<int16_t>(input),
        GetTensorShape(filter), filter_data, GetTensorShape(bias),
        GetTensorData<int32_t>(bias), GetTensorShape(output),
        input->params.scale, output->params.scale, filter_scales,
        GetTensorData<int16_t>(output));
  } else {
    reference_integer_ops::FullyConnectedPerChannel(
        op_params, GetTensorShape(input), GetTensorData<int16_t>(input),
        GetTensorShape(filter), filter_data, GetTensorShape(bias),
        GetTensorData<int64_t>(bias), GetTensorShape(output),
        input->params.scale, output->params.scale, filter_scales,
        GetTensorData<int16_t>(output));
  }
}

}  // namespace

// Verifies that sparsity values are valid given input/weight/output.
bool VerifySparsity(const RuntimeShape& weights_shape,
                    const RuntimeShape& input_shape,
                    const RuntimeShape& output_shape,
                    const TfLiteSparsity* sparsity) {
  const int weights_dims_count = weights_shape.DimensionsCount();
  const int output_dims_count = output_shape.DimensionsCount();
  const int w0_size = sparsity->dim_metadata[0].dense_size;
  const int accum_depth = weights_shape.Dims(weights_dims_count - 1);
  const int output_elements = output_shape.FlatSize();
  const int input_elements = input_shape.FlatSize();
  const int batches = FlatSizeSkipDim(output_shape, output_dims_count - 1);
  const int output_depth = MatchingDim(weights_shape, weights_dims_count - 2,
                                       output_shape, output_dims_count - 1);
  const int max_batch_index = batches - 1;
  const int max_output = max_batch_index * output_depth + w0_size;
  const int max_batch_depth = accum_depth * max_batch_index;

  // Verify output size is enough.
  if (output_elements < max_output) return false;

  // Verify index from sparse in input is valid.
  for (int i = 0; i < sparsity->dim_metadata[1].array_indices->size; ++i) {
    if (input_elements <=
        max_batch_depth + sparsity->dim_metadata[1].array_indices->data[i])
      return false;
  }
  return true;
}

template <KernelType kernel_type>
TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           TfLiteFullyConnectedParams* params, OpData* data,
                           const TfLiteTensor* input,
                           const TfLiteTensor* filter, const TfLiteTensor* bias,
                           TfLiteTensor* output) {
  const bool is_per_channel = data->per_channel_output_multiplier.size() > 1;
  int32_t input_offset = -input->params.zero_point;
  int32_t filter_offset = -filter->params.zero_point;
  int32_t output_offset = output->params.zero_point;
  // Only the Pie path supports quantized models and float inputs/outputs.
  if (input->type == kTfLiteFloat32) {
    TfLiteTensor* input_quantized;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/0,
                                                &input_quantized));
    TfLiteTensor* scaling_factors;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/1,
                                                &scaling_factors));
    TfLiteTensor* accum_scratch;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/2, &accum_scratch));
    TfLiteTensor* input_offsets;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/3, &input_offsets));
    if (data->op_data_4bit) {
      switch (filter->quantization.type) {
        case kTfLiteAffineQuantization:
          return EvalHybridDense4Bit(context, node, params, data, input, filter,
                                     bias, input_quantized, scaling_factors,
                                     accum_scratch, input_offsets, output);
        case kTfLiteBlockwiseQuantization:
          return EvalBlockwise4Bit(context, node, params, data, input, filter,
                                   bias, input_quantized, scaling_factors,
                                   accum_scratch, input_offsets, output);
        default:
          return kTfLiteError;
      }
    }
    TfLiteTensor* row_sums;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, /*index=*/4, &row_sums));
    return EvalHybrid(context, node, params, data, input, filter, bias,
                      input_quantized, scaling_factors, accum_scratch, row_sums,
                      input_offsets, output);
  } else {
    FullyConnectedParams op_params;
    op_params.input_offset = input_offset;
    op_params.weights_offset = filter_offset;
    op_params.output_offset = output_offset;
    op_params.output_multiplier = data->output_multiplier;
    op_params.output_shift = data->output_shift;
    op_params.quantized_activation_min = data->output_activation_min;
    op_params.quantized_activation_max = data->output_activation_max;
    op_params.lhs_cacheable = IsConstantTensor(filter);
    op_params.rhs_cacheable = IsConstantTensor(input);
    if (data->low_precision_applicable){
      int8_t* input_data  = const_cast<int8_t*>(GetTensorData<int8_t>(input));
      int8_t* output_data = GetTensorData<int8_t>(output);

      const int output_dim_count = GetTensorShape(output).DimensionsCount();
      const int filter_dim_count = GetTensorShape(filter).DimensionsCount();
      const int n_batch = FlatSizeSkipDim(GetTensorShape(output), output_dim_count - 1);
      const int n_output = GetTensorShape(filter).Dims(filter_dim_count - 2);
      const int n_input = GetTensorShape(filter).Dims(filter_dim_count - 1);

      int input_size = n_input,
          num_units =  n_output,
          batch_size = n_batch;
      LowPrecision::Method method = LowPrecision::FullyConnected::get_default_method();
      int _kernel_shape[2] = { input_size, num_units  },
          _input_shape[2]  = { batch_size, input_size },
          _output_shape[2] = { batch_size, num_units  };
      
      LowPrecision::Shape kernel_shape = LowPrecision::get_shape(_kernel_shape, 2),
                          input_shape,
                          output_shape;
      
      if (batch_size > 1 || !(LowPrecision::GEMMSupport(method) & LowPrecision::GEMMType::SupportsGEMV)){
        input_shape  = LowPrecision::get_shape(_input_shape,  2);
        output_shape = LowPrecision::get_shape(_output_shape, 2);
      }
      else{
        _input_shape[0] = input_size;
        _output_shape[0] = num_units;
        input_shape  = LowPrecision::get_shape(_input_shape,  1);
        output_shape = LowPrecision::get_shape(_output_shape, 1);
      }
      
      // Creating Timing Profilers
      // LowPrecision::TimingDetailes* input_preprocess_timing_profiler  = new LowPrecision::TimingDetailes();
      // LowPrecision::TimingDetailes* output_preprocess_timing_profiler = new LowPrecision::TimingDetailes();
      // input_preprocess_timing_profiler  ->activate();
      // output_preprocess_timing_profiler ->activate();


      // Getting Kernel Temporary Tensors
      TfLiteTensor* filter_tensor = GetTemporary(context, node, /*index=*/data->filter_temps_idx);

      // Getting Input Temporary Tensors
      std::vector<TfLiteTensor*> input_scratchpads(data->input_temps, nullptr);
      for (size_t i = 0; i < data->input_temps; i++)
        input_scratchpads[i] = GetTemporary(context, node, /*index=*/data->input_temps_idx + i);

      // Creating Input Matrix
      LowPrecision::Matrix input_matrix;
      input_matrix.setData(input_data);
      input_matrix.setShape(input_shape);
      if (data->input_temps >= 1){
        input_matrix.setScratchpad(GetTensorData<int8_t>(input_scratchpads[0]));
        input_matrix.setNeedScratchpad();
      }
      if (data->input_temps >= 2){
        input_matrix.setPaddingScratchpad(GetTensorData<int8_t>(input_scratchpads[1]));
        input_matrix.setPaddingScratchpadSetting();
      }
      input_matrix.setMemLayout(LowPrecision::MemLayout::kRowMajor);

      // Preparing Input Matrix
      LowPrecision::Status input_preparation_status;
      input_preparation_status = LowPrecision::PrepareMatrixAsInputForMethod(input_matrix, method, data->timing_details);
      TF_LITE_ASSERT_EQ(LowPrecision::mask_out_source(LowPrecision::report_on_failure(input_preparation_status, data->low_precision_id, "FC")), LowPrecision::Status::Success);
      
      // Getting Output Temporary Tensors
      TfLiteTensor** output_scratchpads = nullptr;
      output_scratchpads = new TfLiteTensor*[data->output_temps];
      for (size_t i = 0; i < data->output_temps; i++)
        output_scratchpads[i] = GetTemporary(context, node, /*index=*/data->output_temps_idx + i);

      // Creating Output Matrix
      LowPrecision::Matrix output_matrix;
      output_matrix.setData(output_data);
      output_matrix.setShape(output_shape);
      if (data->output_temps >= 1){
        if (LowPrecision::FullyConnected::OutputPreProcess(method) & LowPrecision::PreprocessType::Packing){
          output_matrix.setScratchpad(GetTensorData<int32_t>(output_scratchpads[0]));
          output_matrix.setNeedScratchpad();
        } else {
          output_matrix.setPaddingScratchpad(GetTensorData<int32_t>(output_scratchpads[0]));
          output_matrix.setPaddingScratchpadSetting();
        }
      }
      if (data->output_temps >= 2){
        output_matrix.setPaddingScratchpad(GetTensorData<int32_t>(output_scratchpads[1]));
        output_matrix.setPaddingScratchpadSetting();
      }
      output_matrix.setDowncastCoeff(data->output_multiplier);
      output_matrix.setMemLayout(LowPrecision::MemLayout::kRowMajor);

      // Preparing Output Matrix
      LowPrecision::Status output_preparation_status;
      output_preparation_status = LowPrecision::PrepareMatrixAsOutputForMethod(output_matrix, method, data->timing_details);
      TF_LITE_ASSERT_EQ(LowPrecision::mask_out_source(LowPrecision::report_on_failure(output_preparation_status, data->low_precision_id, "FC")), LowPrecision::Status::Success);

      LowPrecision::Status gemm_status;
      gemm_status = LowPrecision::GEMM(input_matrix, *data->filter_matrix, output_matrix, method, data->timing_details);
      TF_LITE_ASSERT_EQ(LowPrecision::mask_out_source(LowPrecision::report_on_failure(gemm_status, data->low_precision_id, "FC")), LowPrecision::Status::Success);
    }
    else {
      switch (output->type) {
        case kTfLiteUInt8:
          if (kernel_type == kReference) {
            TF_LITE_ENSURE(context, filter->type != kTfLiteInt4);
            reference_ops::FullyConnected(
                op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
                GetTensorShape(filter), GetTensorData<uint8_t>(filter),
                GetTensorShape(bias), GetTensorData<int32_t>(bias),
                GetTensorShape(output), input->params.scale, output->params.scale,
                filter->params.scale, GetTensorData<uint8_t>(output));
          } else {
            optimized_ops::FullyConnected(
                op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
                GetTensorShape(filter), GetTensorData<uint8_t>(filter),
                GetTensorShape(bias), GetTensorData<int32_t>(bias),
                GetTensorShape(output), GetTensorData<uint8_t>(output),
                CpuBackendContext::GetFromContext(context));
          }
          break;
        case kTfLiteInt8:
          if (filter->sparsity != nullptr) {
            const TfLiteSparsity& sparsity = *filter->sparsity;
            const auto input_shape = GetTensorShape(input);
            const auto filter_shape = GetTensorShape(filter);
            const auto output_shape = GetTensorShape(output);
            const auto bias_shape = GetTensorShape(bias);
            if (filter_offset != 0) {
              TF_LITE_KERNEL_LOG(context,
                                "Quantized and sparse fully-connected format "
                                "supports symmetric weight quantization only.");
              return kTfLiteError;
            }
            if (!SupportedSparsityFormat(sparsity) ||
                !VerifySparsity(filter_shape, input_shape, output_shape,
                                &sparsity)) {
              TF_LITE_KERNEL_LOG(
                  context,
                  "Invalid quantized and sparse fully-connected format.");
              return kTfLiteError;
            }
            // Int4 support for sparse filter tensor is currently not supported
            TF_LITE_ENSURE(context, filter->type != kTfLiteInt4);
            if (sparsity.dim_metadata_size == kDimMetadataSizeBlockSparse &&
                sparsity.dim_metadata[2].dense_size == 16) {
              // Block sparse with block size of 1x16.
              optimized_ops::FullyConnectedSparseWeight1x16(
                  sparsity, op_params, input_shape, GetTensorData<int8_t>(input),
                  filter_shape, GetTensorData<int8_t>(filter),
                  data->per_channel_output_multiplier.data(),
                  data->per_channel_output_shift.data(), bias_shape,
                  GetTensorData<int32_t>(bias), output_shape,
                  GetTensorData<int8_t>(output),
                  CpuBackendContext::GetFromContext(context));
            } else {
              TF_LITE_KERNEL_LOG(
                  context, "Unsupported sparse fully-connected weight format.");
              return kTfLiteError;
            }
          } else {
            bool always_false = false;
            const int8_t* filter_data;
            std::unique_ptr<int8_t[]> unpacked_filter_data = nullptr;
            if (filter->type == kTfLiteInt4) {
              const size_t bytes_unpacked = filter->bytes * 2;
              unpacked_filter_data = std::make_unique<int8_t[]>(bytes_unpacked);
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  GetTensorData<int8_t>(filter),
                  GetTensorShape(filter).FlatSize(), unpacked_filter_data.get());
              filter_data = unpacked_filter_data.get();
            } else {
              filter_data = GetTensorData<int8_t>(filter);
            }
            is_per_channel ? FullyConnectedPerChannelInt8<kernel_type>(
                                data, input, filter, filter_data, bias, output,
                                CpuBackendContext::GetFromContext(context))
                          : FullyConnectedInt8<kernel_type>(
                                data, input, filter, filter_data, bias, output,
                                &always_false, NULL, NULL,
                                CpuBackendContext::GetFromContext(context));
          }
          break;
        case kTfLiteInt16:
          if (input->type == kTfLiteInt16) {
            // To avoid 32bit accum overflow, it enables RUY only
            // when zero_point is 0.
            bool has_non_zero_point = input->params.zero_point ||
                                      filter->params.zero_point ||
                                      output->params.zero_point;

            const int8_t* filter_data;
            std::unique_ptr<int8_t[]> unpacked_filter_data = nullptr;
            if (filter->type == kTfLiteInt4) {
              const size_t bytes_unpacked = filter->bytes * 2;
              unpacked_filter_data = std::make_unique<int8_t[]>(bytes_unpacked);
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  GetTensorData<int8_t>(filter),
                  GetTensorShape(filter).FlatSize(), unpacked_filter_data.get());
              filter_data = unpacked_filter_data.get();
            } else {
              filter_data = GetTensorData<int8_t>(filter);
            }

            if (kernel_type == kReference || has_non_zero_point ||
                (bias && bias->type == kTfLiteInt64)) {
              is_per_channel
                  ? FullyConnectedPerChannelInt16<kernel_type>(
                        data, input, filter, filter_data, bias, output)
                  : FullyConnectedInt16<kernel_type>(data, input, filter,
                                                    filter_data, bias, output);
            } else {
              is_per_channel
                  ? optimized_integer_ops::FullyConnectedPerChannel(
                        op_params, data->per_channel_output_multiplier.data(),
                        data->per_channel_output_shift.data(),
                        GetTensorShape(input), GetTensorData<int16_t>(input),
                        GetTensorShape(filter), filter_data, GetTensorShape(bias),
                        GetTensorData<int32_t>(bias), GetTensorShape(output),
                        GetTensorData<int16_t>(output),
                        CpuBackendContext::GetFromContext(context))
                  : optimized_integer_ops::FullyConnected(
                        op_params, GetTensorShape(input),
                        GetTensorData<int16_t>(input), GetTensorShape(filter),
                        filter_data, GetTensorShape(bias),
                        GetTensorData<int32_t>(bias), GetTensorShape(output),
                        GetTensorData<int16_t>(output),
                        CpuBackendContext::GetFromContext(context));
            }
          } else if (kernel_type == kReference) {
            reference_ops::FullyConnected(
                op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
                GetTensorShape(filter), GetTensorData<uint8_t>(filter),
                GetTensorShape(bias), GetTensorData<int32_t>(bias),
                GetTensorShape(output), input->params.scale, output->params.scale,
                filter->params.scale, GetTensorData<int16_t>(output));
          } else {
            optimized_ops::FullyConnected(
                op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
                GetTensorShape(filter), GetTensorData<uint8_t>(filter),
                GetTensorShape(bias), GetTensorData<int32_t>(bias),
                GetTensorShape(output), GetTensorData<int16_t>(output),
                CpuBackendContext::GetFromContext(context));
          }
          break;
        default:
          TF_LITE_KERNEL_LOG(context,
                            "Quantized FullyConnected expects output data "
                            "type uint8, int8 or int16");
          return kTfLiteError;
      }
    }
  }

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalShuffledQuantized(TfLiteContext* context, TfLiteNode* node,
                                   TfLiteFullyConnectedParams* params,
                                   OpData* data, const TfLiteTensor* input,
                                   const TfLiteTensor* filter,
                                   const TfLiteTensor* bias,
                                   TfLiteTensor* output,
                                   TfLiteTensor* shuffled_input_workspace) {
  // TODO(b/110697972) decide more consistently if / how / where we want
  // to perform this kind of runtime data type checks.
  if (shuffled_input_workspace->type != kTfLiteUInt8) {
    TF_LITE_KERNEL_LOG(context, "Unexpected data type");
    return kTfLiteError;
  }

#define TF_LITE_SHUFFLED_FULLY_CONNECTED(type)                           \
  {                                                                      \
    type::ShuffledFullyConnected(                                        \
        op_params, GetTensorShape(input), GetTensorData<uint8_t>(input), \
        GetTensorShape(filter), GetTensorData<uint8_t>(filter),          \
        GetTensorShape(bias), GetTensorData<int32_t>(bias),              \
        GetTensorShape(output), GetTensorData<int16_t>(output),          \
        GetTensorData<uint8_t>(shuffled_input_workspace),                \
        CpuBackendContext::GetFromContext(context));                     \
  }
  FullyConnectedParams op_params;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  op_params.lhs_cacheable = IsConstantTensor(filter);
  op_params.rhs_cacheable = IsConstantTensor(input);
  if (kernel_type == kReference) {
    reference_ops::ShuffledFullyConnected(
        op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
        GetTensorShape(filter), GetTensorData<uint8_t>(filter),
        GetTensorShape(bias), GetTensorData<int32_t>(bias),
        GetTensorShape(output), GetTensorData<int16_t>(output),
        GetTensorData<uint8_t>(shuffled_input_workspace));
  } else {
    optimized_ops::ShuffledFullyConnected(
        op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
        GetTensorShape(filter), GetTensorData<uint8_t>(filter),
        GetTensorShape(bias), GetTensorData<int32_t>(bias),
        GetTensorShape(output), GetTensorData<int16_t>(output),
        GetTensorData<uint8_t>(shuffled_input_workspace),
        CpuBackendContext::GetFromContext(context));
  }
#undef TF_LITE_SHUFFLED_FULLY_CONNECTED

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       TfLiteFullyConnectedParams* params, OpData* data,
                       const TfLiteTensor* input, const TfLiteTensor* filter,
                       const TfLiteTensor* bias, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);
  if (kernel_type == kReference) {
    FullyConnectedParams op_params;
    op_params.float_activation_min = output_activation_min;
    op_params.float_activation_max = output_activation_max;
    if (filter->sparsity != nullptr) {
      const auto& sparsity = *filter->sparsity;
      reference_ops::FullyConnectedSparseWeight(
          sparsity, op_params, GetTensorShape(input),
          GetTensorData<float>(input), GetTensorShape(filter),
          GetTensorData<float>(filter), GetTensorShape(bias),
          GetTensorData<float>(bias), GetTensorShape(output),
          GetTensorData<float>(output));
    } else {
      reference_ops::FullyConnected(
          op_params, GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(filter), GetTensorData<float>(filter),
          GetTensorShape(bias), GetTensorData<float>(bias),
          GetTensorShape(output), GetTensorData<float>(output));
    }
  } else if (kernel_type == kLegacyPie) {
    return EvalPie(context, node, params, data, input, filter, bias, output);
  } else {
    FullyConnectedParams op_params;
    op_params.float_activation_min = output_activation_min;
    op_params.float_activation_max = output_activation_max;
    if (filter->sparsity != nullptr) {
      const auto& sparsity = *filter->sparsity;
      if (!SupportedSparsityFormat(sparsity)) {
        TF_LITE_KERNEL_LOG(context,
                           "Unsupported sparse fully-connected weight format.");
        return kTfLiteError;
      }
      const auto& input_shape = GetTensorShape(input);
      const auto& filter_shape = GetTensorShape(filter);
      const auto& output_shape = GetTensorShape(output);
      const auto& bias_shape = GetTensorShape(bias);
      if (!VerifySparsity(filter_shape, input_shape, output_shape, &sparsity)) {
        TF_LITE_KERNEL_LOG(context, "Invalid sparse fully-connected format.");
        return kTfLiteError;
      }

      if (sparsity.dim_metadata_size == kDimMetadataSizeRandomSparse) {
        // Random sparse.
        optimized_ops::FullyConnectedSparseWeight(
            sparsity, op_params,                         // Disable formatting
            input_shape, GetTensorData<float>(input),    // Disable formatting
            filter_shape, GetTensorData<float>(filter),  // Disable formatting
            bias_shape, GetTensorData<float>(bias),      // Disable formatting
            output_shape, GetTensorData<float>(output));
      } else if (sparsity.dim_metadata_size == kDimMetadataSizeBlockSparse &&
                 sparsity.dim_metadata[2].dense_size == 4) {
        // Block sparse with block size of 1x4.
        optimized_ops::FullyConnectedSparseWeight1x4(
            sparsity, op_params,                         // Disable formatting
            input_shape, GetTensorData<float>(input),    // Disable formatting
            filter_shape, GetTensorData<float>(filter),  // Disable formatting
            bias_shape, GetTensorData<float>(bias),      // Disable formatting
            output_shape, GetTensorData<float>(output),
            CpuBackendContext::GetFromContext(context));
      } else {
        TF_LITE_KERNEL_LOG(context,
                           "Unsupported sparse fully-connected weight format.");
        return kTfLiteError;
      }

    } else {
      op_params.lhs_cacheable = IsConstantTensor(filter);
      op_params.rhs_cacheable = IsConstantTensor(input);
      optimized_ops::FullyConnected(
          op_params, GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(filter), GetTensorData<float>(filter),
          GetTensorShape(bias), GetTensorData<float>(bias),
          GetTensorShape(output), GetTensorData<float>(output),
          CpuBackendContext::GetFromContext(context));
    }
  }

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* filter;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kWeightsTensor, &filter));
  const TfLiteTensor* bias =
      (node->inputs->size == 3)
          ? GetOptionalInputTensor(context, node, kBiasTensor)
          : nullptr;
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  // Do nothing if expected output is empty.
  if (NumElements(output) == 0) {
    return kTfLiteOk;
  }

  if (filter->dims->data[1] == 0) {
    memset(output->data.data, 0, output->bytes);
    return kTfLiteOk;
  }

  switch (filter->type) {
    case kTfLiteFloat32:
      return EvalFloat<kernel_type>(context, node, params, data, input, filter,
                                    bias, output);
    case kTfLiteUInt8:
      if (params->weights_format ==
          kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8) {
        TfLiteTensor* shuffled_input_workspace;
        TF_LITE_ENSURE_OK(
            context, GetOutputSafe(context, node, kShuffledInputWorkspaceTensor,
                                   &shuffled_input_workspace));
        return EvalShuffledQuantized<kernel_type>(context, node, params, data,
                                                  input, filter, bias, output,
                                                  shuffled_input_workspace);
      } else if (params->weights_format ==
                 kTfLiteFullyConnectedWeightsFormatDefault) {
        return EvalQuantized<kernel_type>(context, node, params, data, input,
                                          filter, bias, output);
      } else {
        TF_LITE_KERNEL_LOG(context, "Unhandled fully-connected weights format");
        return kTfLiteError;
      }
    case kTfLiteInt8:
      if (params->weights_format == kTfLiteFullyConnectedWeightsFormatDefault) {
        return EvalQuantized<kernel_type>(context, node, params, data, input,
                                          filter, bias, output);
      } else {
        TF_LITE_KERNEL_LOG(context, "Unhandled fully-connected weights format");
        return kTfLiteError;
      }
    case kTfLiteInt4:
      if (params->weights_format == kTfLiteFullyConnectedWeightsFormatDefault) {
        return EvalQuantized<kernel_type>(context, node, params, data, input,
                                          filter, bias, output);
      } else {
        TF_LITE_KERNEL_LOG(context, "Unhandled fully-connected weights format");
        return kTfLiteError;
      }
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Filter data type %s currently not supported.",
                         TfLiteTypeGetName(filter->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace fully_connected

TfLiteRegistration* Register_FULLY_CONNECTED_REF() {
  static TfLiteRegistration r = {
      fully_connected::Init, fully_connected::Free,
      fully_connected::Prepare<fully_connected::kReference>,
      fully_connected::Eval<fully_connected::kReference>};
  return &r;
}

TfLiteRegistration* Register_FULLY_CONNECTED_GENERIC_OPT() {
  static TfLiteRegistration r = {
      fully_connected::Init, fully_connected::Free,
      fully_connected::Prepare<fully_connected::kGenericOptimized>,
      fully_connected::Eval<fully_connected::kGenericOptimized>};
  return &r;
}

// Legacy path for PIE clients.
TfLiteRegistration* Register_FULLY_CONNECTED_PIE() {
  static TfLiteRegistration r = {
      fully_connected::Init, fully_connected::Free,
      fully_connected::Prepare<fully_connected::kLegacyPie>,
      fully_connected::Eval<fully_connected::kLegacyPie>};
  return &r;
}

TfLiteRegistration* Register_FULLY_CONNECTED() {
  return Register_FULLY_CONNECTED_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

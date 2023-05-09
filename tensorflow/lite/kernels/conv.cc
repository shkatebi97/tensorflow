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
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/conv.h"

#include <stddef.h>

#include <cstdint>
#include <vector>

// Only use multi-threaded Eigen if ruy is disabled.
#if !defined(TFLITE_WITH_RUY)
#define TFLITE_WITH_MULTITHREADED_EIGEN
#endif

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#if defined(TFLITE_WITH_MULTITHREADED_EIGEN)
#include "tensorflow/lite/kernels/eigen_support.h"
#endif
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/types.h"
// b/131835803 forces us to include multithreaded_conv.h before optimized_ops.h
#if defined(TFLITE_WITH_MULTITHREADED_EIGEN)
#include "tensorflow/lite/kernels/internal/optimized/multithreaded_conv.h"
#endif
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/optimized-low-precision/common/types.h"
#include "tensorflow/lite/kernels/optimized-low-precision/low_precision_fully_connected.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace conv {

// This file has 4 implementation of Conv.
enum KernelType {
  kReference,
  kGenericOptimized,  // Neon-free
  // kMultithreadOptimized is a mixture of an Eigen-based kernel when threads
  // are available and kGenericOptimized when we must use only one thread.
  kMultithreadOptimized,
  // The kernel uses use CBLAS interface for matrix multiplication.
  // It's fast when an optimized CBLAS implementation is available (e.g. Apple
  // Accelerate Framework), and it's slow when falling back to naive
  // implementation.
  kCblasOptimized,
};

const int kTensorNotAllocated = -1;

static constexpr size_t kMaxIm2colBufferSizeMobile = 1024 * 1024 * 1024;  // 1GB

struct OpData {
  // IDs are the arbitrary identifiers used by TF Lite to identify and access
  // memory buffers.
  int im2col_id = kTensorNotAllocated;
  int hwcn_weights_id = kTensorNotAllocated;
  int input_quantized_id = kTensorNotAllocated;
  int scaling_factors_id = kTensorNotAllocated;
  int input_offset_id = kTensorNotAllocated;
  int accum_scratch_id = kTensorNotAllocated;
  // Row sums are used to cache filter sums for hybrid zero-point calculations.
  int row_sums_id = kTensorNotAllocated;

  TfLitePaddingValues padding;
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
  // Indexes are the offset to the memory buffer in the array used to keep track
  // of the allocated temporaries.
  int32_t im2col_index;
  int32_t hwcn_weights_index;
  int32_t input_quantized_index;
  int32_t scaling_factors_index;
  int32_t accum_scratch_index;
  int32_t input_offset_index;
  int32_t row_sums_index;

  bool need_hwcn_weights = false;
  bool have_weights_been_transposed = false;
  bool need_im2col = false;
  // If it's true, it means im2col is needed but gets disabled because the
  // temporary im2col tensor requires too much memory (i.e.
  // >= kMaxIm2colBufferSize);
  bool im2col_oversized = false;

  bool supports_multithreaded_kernel = false;
  bool is_hybrid_per_channel = false;
  bool compute_hybrid_row_sums = true;

  // long int low_precision_id = 0;
  // bool low_precision_applicable = false;
  // bool low_precision_activation_applicable = false;
  int32_t low_precision_weight_index;
  int32_t low_precision_activation_index;
  // int low_precision_weight_id = kTensorNotAllocated;
  // int low_precision_activation_id = kTensorNotAllocated;
  long int low_precision_id = 0;
  bool low_precision_activation_applicable = false;
  bool low_precision_applicable = false;
  bool low_precision_multibatched = false;
  bool low_precision_compress_activation = false;
  LowPrecision::Method operation_method = LowPrecision::Method::kNoOptimization;
  int kernel_temps;
  int input_temps;
  int output_temps;
  int kernel_temps_idx;
  int input_temps_idx;
  int output_temps_idx;
  int filter_temps_idx;
  std::vector<int> kernel_temps_id;
  std::vector<int> input_temps_id;
  std::vector<int> output_temps_id;
  int filter_temps_id = kTensorNotAllocated;
  LowPrecision::Matrix* filter_matrix = nullptr;
  LowPrecision::TimingDetailes* timing_details = nullptr;
};

struct ULP_Params{
  LowPrecision::Shape* input_shape;
  LowPrecision::Shape* input_shape_w_im2col;
  LowPrecision::Shape* input_shape_wo_im2col;
  LowPrecision::Shape* filter_shape;
  LowPrecision::Shape* output_shape;
  LowPrecision::Method method;
  LowPrecision::DataType input_type;
  LowPrecision::DataType filter_type;
  LowPrecision::DataType output_type;
  LowPrecision::ShapeList kernel_scratchpads_shape_list;
  LowPrecision::ShapeList input_scratchpads_shape_list;
  LowPrecision::ShapeList output_scratchpads_shape_list;
};

inline PaddingType RuntimePaddingType(TfLitePadding padding) {
  switch (padding) {
    case TfLitePadding::kTfLitePaddingSame:
      return PaddingType::kSame;
    case TfLitePadding::kTfLitePaddingValid:
      return PaddingType::kValid;
    case TfLitePadding::kTfLitePaddingUnknown:
    default:
      return PaddingType::kNone;
  }
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to use as scratch space for im2col, and
  // to carry information from Prepare() to Eval().
  auto* data = new OpData;
#if defined(TFLITE_WITH_MULTITHREADED_EIGEN)
  eigen_support::IncrementUsageCounter(context);
#endif
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
#if defined(TFLITE_WITH_MULTITHREADED_EIGEN)
  eigen_support::DecrementUsageCounter(context);
#endif
  OpData* data = reinterpret_cast<OpData*>(buffer);
  if (data->low_precision_applicable){
    // std::cout << "GEMM API Timing (CONV-" << data->low_precision_id 
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

// Naive implementation of transpose for floats. Could be optimized to be more
// cache friendly, but for now it's a one-time cost on first run, and we would
// prefer to remove the need to do this at all eventually.
void TransposeFloatTensor(const TfLiteTensor* input, TfLiteTensor* output) {
  const int rows = output->dims->data[1];
  const int cols = output->dims->data[0];
  const float* input_data = GetTensorData<float>(input);
  float* output_data = GetTensorData<float>(output);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const float in_value = input_data[i * cols + j];
      output_data[j * rows + i] = in_value;
    }
  }
}

// Check if im2col needs to be allocated, as some version of optimized Conv dont
// use it. If any change is supporting im2col in any of the Conv versions, then
// it should be updated here as well
bool IsIm2ColRequired(const TfLiteTensor* input, TfLiteConvParams* params,
                      const TfLiteTensor* filter, OpData* data, bool is_hybrid,
                      KernelType kernel_type) {
  // If HWCN weights are required, Im2Col not required
  if (data->need_hwcn_weights) return false;

  // segregate based on dilated conv & non-dialated conv
  const bool need_dilated_im2col =
      params->dilation_width_factor != 1 || params->dilation_height_factor != 1;
  const bool need_non_dilated_im2col =
      params->stride_width != 1 || params->stride_height != 1 ||
      filter->dims->data[2] != 1 || filter->dims->data[1] != 1;

  const bool need_im2col = need_dilated_im2col || need_non_dilated_im2col;

  // Return early as basic requirement is not met
  if (!need_im2col) return false;

  // Special case for Hybrid, as it supports only non-dilated im2col currently
  const bool is_hybrid_non_dilated = is_hybrid && need_non_dilated_im2col;
  const bool is_quantized =
      input->type == kTfLiteUInt8 || input->type == kTfLiteInt8;

  switch (kernel_type) {
    case kReference:
      if (is_hybrid) {
        return true;
      } else {
        return false;
      }
    case kGenericOptimized:
    case kCblasOptimized:
      if (is_hybrid && !need_non_dilated_im2col) {
        return false;
      } else {
        return true;
      }
    case kMultithreadOptimized:
      if (is_hybrid_non_dilated || is_quantized ||
          !data->supports_multithreaded_kernel) {
        return true;
      } else {
        return false;
      }
    default:
      return false;
  }
}

// Allocate temporary tensors (`im2col`, `hwcn_weights` if necessary).
// Note: `context->AddTensors` might invalidate pointers to existing tensors.
// Therefore the logic to add tensors are isolated into this function.
static TfLiteStatus AllocateTemporaryTensorsIfRequired(
    TfLiteContext* context, TfLiteNode* node, bool is_hybrid,
    bool is_per_channel, KernelType kernel_type, size_t im2col_bytes,
    ULP_Params* ulp_params) {
  auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE(context, node->inputs->size >= 2);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  const TfLiteTensor* filter;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &filter));

  // If we're using the optimized multithreaded EigenTensor implementation of
  // convolution, it expects the filter weights to be transposed compared to
  // the normal TF Lite buffer format. Typical TF Lite weights are
  // [filter_count, filter_height, filter_width, input_depth], but for the float
  // implementation we need them as [filter_height, filter_width, input_depth,
  // filter_count]. We get to that format by transposing, and create a temporary
  // buffer to store the results.
  // This path is only used for float processing, so only create the buffer if
  // we're running with that data type.
  data->need_hwcn_weights =
      input->type == kTfLiteFloat32 && data->supports_multithreaded_kernel;

  // We don't always need to allocate im2col. It is only used in some versions
  // of the optimized Conv. This test just mimics something that happens inside
  // optimized_ops.h, in order to avoid a DCHECK(!im2col_data).
  data->need_im2col =
      IsIm2ColRequired(input, params, filter, data, is_hybrid, kernel_type);

  // If im2col_oversized is found to be true, we have to fallback to an
  // execution path (like kReference in float/quantized cases) that doesn't
  // require im2col operation. Therefore, we have to skip checking the hybrid
  // case (but not the hybrid-per-channel one) where there's no such a fallback
  // execution path.
  // TODO(b/178743262): Consider making this check conditioned on the available
  // memory of the system, rather than coupling to the mobile platform check.
  if (IsMobilePlatform() && !(is_hybrid && !is_per_channel) &&
      data->need_im2col && im2col_bytes >= kMaxIm2colBufferSizeMobile) {
    data->need_im2col = false;
    data->im2col_oversized = true;
  }
  int temporaries_count = 0;
  if (data->need_im2col) {
    data->im2col_index = temporaries_count;
    if (data->im2col_id == kTensorNotAllocated) {
      context->AddTensors(context, 1, &data->im2col_id);
    }
    ++temporaries_count;
  }
  if (data->need_hwcn_weights) {
    data->hwcn_weights_index = temporaries_count;
    if (data->hwcn_weights_id == kTensorNotAllocated) {
      context->AddTensors(context, 1, &data->hwcn_weights_id);
    }
    ++temporaries_count;
  }

  if (is_hybrid) {
    // Allocate tensor to store the on-the-fly quantized inputs.
    data->input_quantized_index = temporaries_count;
    if (data->input_quantized_id == kTensorNotAllocated) {
      TF_LITE_ENSURE_OK(
          context, context->AddTensors(context, 1, &data->input_quantized_id));
    }
    ++temporaries_count;

    // Allocate tensor to store the quantization params computed during
    // on-the-fly input quantization.
    data->scaling_factors_index = temporaries_count;
    if (data->scaling_factors_id == kTensorNotAllocated) {
      TF_LITE_ENSURE_OK(
          context, context->AddTensors(context, 1, &data->scaling_factors_id));
    }
    ++temporaries_count;

    // Allocate tensor to store the accumulators for the matrix multiply.
    data->accum_scratch_index = temporaries_count;
    if (data->accum_scratch_id == kTensorNotAllocated) {
      TF_LITE_ENSURE_OK(
          context, context->AddTensors(context, 1, &data->accum_scratch_id));
    }
    ++temporaries_count;
    if (is_per_channel) {
      data->input_offset_index = temporaries_count;
      if (data->input_offset_id == kTensorNotAllocated) {
        TF_LITE_ENSURE_OK(
            context, context->AddTensors(context, 1, &data->input_offset_id));
      }
      ++temporaries_count;

      data->row_sums_index = temporaries_count;
      if (data->row_sums_id == kTensorNotAllocated) {
        TF_LITE_ENSURE_OK(context,
                          context->AddTensors(context, 1, &data->row_sums_id));
      }
      ++temporaries_count;
    }
  }

  if (data->need_im2col) {
    ulp_params->input_shape->size[1] = ulp_params->input_shape_w_im2col->size[1];
    ulp_params->input_shape->size[0] = ulp_params->input_shape_w_im2col->size[0];
  } else {
    ulp_params->input_shape->size[1] = ulp_params->input_shape_wo_im2col->size[1];
    ulp_params->input_shape->size[0] = ulp_params->input_shape_wo_im2col->size[0];
  }
  std::cerr << "\tChanging Input Shape" << std::endl;
  std::cerr << "\tNew Input Shape: " << LowPrecision::get_shape_string(*(ulp_params->input_shape))
            << std::endl;
  bool should_apply_low_precision = LowPrecision::FullyConnected::IsAppliable(
      ulp_params->method, *(ulp_params->input_shape), *(ulp_params->filter_shape),
      ulp_params->input_type, ulp_params->filter_type, ulp_params->output_type, 
      true);
  bool includes_low_precision_activation =
      LowPrecision::FullyConnected::IncludesActivationCompression(ulp_params->method) ||
      ulp_params->input_shape->size[0] > 1;
  if (data->low_precision_applicable && !should_apply_low_precision)
    std::cerr << "\tNot Applying Now." << std::endl;
  else if (!data->low_precision_applicable && should_apply_low_precision)
    std::cerr << "\tApplying Now." << std::endl;
  else
    std::cerr << "\tNo Changes To Appliability." << std::endl;
  data->low_precision_applicable = should_apply_low_precision;
  data->low_precision_activation_applicable = includes_low_precision_activation;
  
  if (data->low_precision_applicable) {
    data->operation_method = ulp_params->method;

    ulp_params->kernel_scratchpads_shape_list = LowPrecision::GetFilterShapeListForMethod(ulp_params->method, *ulp_params->filter_shape);
    ulp_params->input_scratchpads_shape_list  = LowPrecision::GetInputShapeListForMethod (ulp_params->method, *ulp_params->input_shape);
    ulp_params->output_scratchpads_shape_list = LowPrecision::GetOutputShapeListForMethod(ulp_params->method, *ulp_params->input_shape, *ulp_params->filter_shape, *ulp_params->output_shape);

    int num_kernel_scratchpads = ulp_params->kernel_scratchpads_shape_list.size(),
        num_input_scratchpads  = ulp_params->input_scratchpads_shape_list.size(),
        num_output_scratchpads = ulp_params->output_scratchpads_shape_list.size();
    
    data->kernel_temps = num_kernel_scratchpads;
    data->input_temps  = num_input_scratchpads;
    data->output_temps = num_output_scratchpads;

    if (data->kernel_temps_id.size() < 1)
      data->kernel_temps_id.resize(num_kernel_scratchpads,  kTensorNotAllocated);
    if (data->input_temps_id.size() < 1)
      data->input_temps_id .resize(num_input_scratchpads,   kTensorNotAllocated);
    if (data->output_temps_id.size() < 1)
      data->output_temps_id.resize(num_output_scratchpads,  kTensorNotAllocated);

    data->filter_temps_idx = temporaries_count;
    data->kernel_temps_idx = data->filter_temps_idx + num_kernel_scratchpads - 1;
    data->input_temps_idx  = data->filter_temps_idx + num_kernel_scratchpads    ;
    data->output_temps_idx = data->input_temps_idx  + num_input_scratchpads     ;

    data->timing_details = new LowPrecision::TimingDetailes();
    data->timing_details->activate(LowPrecision::FullyConnected::GetVariableFromEnv( "GEMMAPITiming_Disable" ) != "TRUE");

    bool k_need_preparing = false,
         i_need_preparing = false,
         o_need_preparing = false;
         
    std::cerr << "\tReserving " << num_kernel_scratchpads +
                  num_input_scratchpads +
                  num_output_scratchpads << " LowPrecision Tensors In Total" << std::endl;


    // Allocating Filter and Required Kernel Scratchpads
    if (num_kernel_scratchpads >= 1){ // Filter Tensor
      if (data->filter_temps_id == kTensorNotAllocated)
        TF_LITE_ENSURE_OK(
            context, context->AddTensors(context, 1, &data->filter_temps_id));
      data->kernel_temps_id[0] = data->filter_temps_id;
      ++temporaries_count;
    }
    for (int i = 1 ; i < num_kernel_scratchpads ; i++){ // Kernel Scratchpads Tensor
      if (data->kernel_temps_id[i] == kTensorNotAllocated)
        TF_LITE_ENSURE_OK(
            context, context->AddTensors(context, 1, &data->kernel_temps_id[i]));
      ++temporaries_count;
    }
    // Allocating Required Input Scratchpads
    for (int i = 0 ; i < num_input_scratchpads ; i++){ // Input Scratchpads Tensor
      if (data->input_temps_id[i] == kTensorNotAllocated)
        TF_LITE_ENSURE_OK(
            context, context->AddTensors(context, 1, &data->input_temps_id[i]));
      ++temporaries_count;
    }
    // Allocating Required Output Scratchpads
    for (int i = 0 ; i < num_output_scratchpads ; i++){ // Output Scratchpads Tensor
      if (data->output_temps_id[i] == kTensorNotAllocated)
        TF_LITE_ENSURE_OK(
            context, context->AddTensors(context, 1, &data->output_temps_id[i]));
      ++temporaries_count;
    }
  }

  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(temporaries_count);

  return kTfLiteOk;
}

TfLiteStatus Prepare(KernelType kernel_type, TfLiteContext* context,
                     TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  bool has_bias = node->inputs->size == 3;
  // Check number of inputs/outputs
  TF_LITE_ENSURE(context, has_bias || node->inputs->size == 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  const TfLiteTensor* filter;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &filter));

  // Check dimensionality of input, filter
  TF_LITE_ENSURE_EQ(context, input->dims->size, 4);
  TF_LITE_ENSURE_EQ(context, filter->dims->size, 4);
  // Check input channels matching filter
  TF_LITE_ENSURE_EQ(context, input->dims->data[3], filter->dims->data[3]);

  // Check types. (We assume that UINT8 refers to quantized tensors)
  TfLiteType input_type = input->type;
  TF_LITE_ENSURE(context,
                 input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8 ||
                     input_type == kTfLiteInt8 || input_type == kTfLiteInt16);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, input_type);

  if (input_type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
  }
  // Filter must have zero zero-points in per-channel quantization.
  if (input_type == kTfLiteInt16 || input_type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    for (int i = 0; i < affine_quantization->zero_point->size; ++i) {
      TF_LITE_ENSURE_EQ(context, affine_quantization->zero_point->data[i], 0);
    }
  }

  const TfLiteTensor* bias = nullptr;

  // TODO(ahentz): At this point the optimized versions require 'bias'. We can
  // either change that or document that convolution requires it.
  TF_LITE_ENSURE(context, has_bias);

  if (has_bias) {
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 2, &bias));
    if (input_type == kTfLiteUInt8 || input_type == kTfLiteInt8) {
      TF_LITE_ENSURE_TYPES_EQ(context, bias->type, kTfLiteInt32);
      TF_LITE_ENSURE_EQ(context, bias->params.zero_point, 0);
    } else if (input_type == kTfLiteInt16) {
      TF_LITE_ENSURE_TYPES_EQ(context, bias->type, kTfLiteInt64);
      TF_LITE_ENSURE_EQ(context, bias->params.zero_point, 0);
    } else {
      TF_LITE_ENSURE_TYPES_EQ(context, bias->type, input_type);
    }
    TF_LITE_ENSURE_EQ(context, NumElements(bias), SizeOfDimension(filter, 0));
  }

  const bool is_hybrid =
      (input->type == kTfLiteFloat32 &&
       (filter->type == kTfLiteUInt8 || filter->type == kTfLiteInt8));

  if (is_hybrid && filter->type == kTfLiteInt8 &&
      filter->quantization.type == kTfLiteAffineQuantization &&
      filter->quantization.params &&
      reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params)
          ->scale &&
      reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params)
              ->scale->size > 1) {
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    const float scale = affine_quantization->scale->data[0];
    for (int i = 1; i < affine_quantization->scale->size; i++) {
      if (affine_quantization->scale->data[i] != scale) {
        data->is_hybrid_per_channel = true;
        break;
      }
    }
  }

  // The multi-threaded kernel supports neither dilation nor hybrid kernels, and
  // is incompatible with mutable input filters that might change between evals.
  data->supports_multithreaded_kernel =
      (kernel_type == kMultithreadOptimized) &&
      (context->recommended_num_threads != 1) && !is_hybrid &&
      (params->dilation_width_factor == 1) &&
      (params->dilation_height_factor == 1) &&
      (filter->allocation_type != kTfLiteArenaRw) && !IsDynamicTensor(filter);

  int channels_in = filter->dims->data[3];
  int channels_out = filter->dims->data[0];
  int width = input->dims->data[2];
  int height = input->dims->data[1];
  int filter_width = filter->dims->data[2];
  int filter_height = filter->dims->data[1];
  int batches = input->dims->data[0];

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params->padding;
  int out_width, out_height;
  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width,
      params->dilation_height_factor, params->dilation_width_factor, height,
      width, filter_height, filter_width, padding, &out_height, &out_width);

  int __filter_shape_sizes[2] = {
      filter->dims->data[0],
      filter->dims->data[1] * filter->dims->data[2] * filter->dims->data[3]};
  int __input_shape_sizes[2];
  int __input_shape_sizes_w_im2col[2];
  int __input_shape_sizes_wo_im2col[2];
  if (data->need_im2col) {
    __input_shape_sizes[1] = input->dims->data[3] * filter_height * filter_width;
    __input_shape_sizes[0] = batches * out_height * out_width;
  } else {
    __input_shape_sizes[1] = input->dims->data[3];
    __input_shape_sizes[0] = input->dims->data[1] * input->dims->data[2] * input->dims->data[0];
  }
  __input_shape_sizes_w_im2col[1] = input->dims->data[3] * filter_height * filter_width;
  __input_shape_sizes_w_im2col[0] = batches * out_height * out_width;
  __input_shape_sizes_wo_im2col[1] = input->dims->data[3];
  __input_shape_sizes_wo_im2col[0] = input->dims->data[1] * input->dims->data[2] * input->dims->data[0];
  int __output_shape_sizes[2] = {
    output->dims->data[0] * output->dims->data[1] * output->dims->data[2],
    output->dims->data[3]
  };
  LowPrecision::Shape __input_shape =
      LowPrecision::get_shape(__input_shape_sizes, 2);
  LowPrecision::Shape __input_shape_w_im2col =
      LowPrecision::get_shape(__input_shape_sizes_w_im2col, 2);
  LowPrecision::Shape __input_shape_wo_im2col =
      LowPrecision::get_shape(__input_shape_sizes_wo_im2col, 2);
  LowPrecision::Shape __filter_shape =
      LowPrecision::get_shape(__filter_shape_sizes, 2);
  __filter_shape = __filter_shape.T();
  LowPrecision::Shape __output_shape =
      LowPrecision::get_shape(__output_shape_sizes, 2);

  LowPrecision::Method __method =
      LowPrecision::FullyConnected::GetMethodFromEnv();

  bool should_apply_low_precision = LowPrecision::FullyConnected::IsAppliable(
      __method, __input_shape, __filter_shape,
      LowPrecision::FullyConnected::GetDataType(input->type),
      LowPrecision::FullyConnected::GetDataType(filter->type),
      LowPrecision::FullyConnected::GetDataType(output->type), true);
  bool includes_low_precision_activation =
      LowPrecision::FullyConnected::IncludesActivationCompression(__method) ||
      __input_shape.size[0] > 1;
  ULP_Params ulp_params;
  ulp_params.input_shape = &__input_shape;
  ulp_params.input_shape_w_im2col = &__input_shape_w_im2col;
  ulp_params.input_shape_wo_im2col = &__input_shape_wo_im2col;
  ulp_params.filter_shape = &__filter_shape;
  ulp_params.output_shape = &__output_shape;
  ulp_params.method = __method;
  ulp_params.input_type = LowPrecision::FullyConnected::GetDataType(input->type);
  ulp_params.filter_type = LowPrecision::FullyConnected::GetDataType(filter->type);
  ulp_params.output_type = LowPrecision::FullyConnected::GetDataType(output->type);
  data->low_precision_id = LowPrecision::FullyConnected::id++;
  // std::cout << "input_shape_w_im2col: " << LowPrecision::get_shape_string(__input_shape_w_im2col) << std::endl;
  // std::cout << "input_shape_wo_im2col: " << LowPrecision::get_shape_string(__input_shape_wo_im2col) << std::endl;
  // std::cout << "Need_Im2col-b: " << data->need_im2col << std::endl;
  if (should_apply_low_precision)
    std::cerr << "Applying Conv Low-Precision for Kernel shape "
              << LowPrecision::get_shape_string(__filter_shape)
              << ", Input shape "
              << LowPrecision::get_shape_string(__input_shape)
              << ", and Output shape "
              << LowPrecision::get_shape_string(__output_shape)
              << ", and the ID is "
              << data->low_precision_id
              << std::endl;
  else
    std::cerr << "NOT Applying Conv Low-Precision for Kernel shape "
              << LowPrecision::get_shape_string(__filter_shape)
              << ", Input shape "
              << LowPrecision::get_shape_string(__input_shape)
              << ", and Output shape "
              << LowPrecision::get_shape_string(__output_shape)
              << ", and the ID is "
              << data->low_precision_id
              << std::endl;

  data->low_precision_applicable = should_apply_low_precision;
  data->low_precision_activation_applicable = includes_low_precision_activation;

  size_t im2col_type_size;
  TF_LITE_ENSURE_STATUS(GetSizeOfType(context, input->type, &im2col_type_size));
  // Note that we intentionally promote the first multiplicand (i.e. 'batches')
  // to 'size_t' to avoid integer overflow here.
  const size_t im2col_bytes = static_cast<size_t>(batches) * out_height *
                              out_width * channels_in * filter_height *
                              filter_width * im2col_type_size;
  TF_LITE_ENSURE_STATUS(AllocateTemporaryTensorsIfRequired(
      context, node, is_hybrid, data->is_hybrid_per_channel, kernel_type,
      im2col_bytes, &ulp_params));
  
  // std::cout << "Need_Im2col-a: " << data->need_im2col << std::endl;

  TF_LITE_ENSURE(context, has_bias);

  // Note that full fixed-point inference requires that all tensors have their
  // parameters set. This is usually done during quantized training or
  // calibration.
  if (input_type != kTfLiteFloat32) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    TF_LITE_ENSURE(context, (affine_quantization->scale->size == 1 ||
                             affine_quantization->scale->size == channels_out));

    data->per_channel_output_multiplier.resize(channels_out);
    data->per_channel_output_shift.resize(channels_out);
    TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier.data(),
        data->per_channel_output_shift.data(), channels_out));
  }

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = batches;
  output_size->data[1] = out_height;
  output_size->data[2] = out_width;
  output_size->data[3] = channels_out;
  auto output_status = context->ResizeTensor(context, output, output_size);

  if (output_status != kTfLiteOk) return output_status;

  if (data->need_im2col) {
    node->temporaries->data[data->im2col_index] = data->im2col_id;

    TfLiteIntArray* im2col_size = TfLiteIntArrayCreate(4);

    int input_depth = input->dims->data[3];
    im2col_size->data[0] = output_size->data[0];
    im2col_size->data[1] = output_size->data[1];
    im2col_size->data[2] = output_size->data[2];
    im2col_size->data[3] = input_depth * filter_height * filter_width;

    TfLiteTensor* im2col =
        &context->tensors[node->temporaries->data[data->im2col_index]];
    im2col->type = input->type;
    if (is_hybrid) {
      im2col->type = filter->type;
    }
    im2col->allocation_type = kTfLiteArenaRw;
    auto im2col_status = context->ResizeTensor(context, im2col, im2col_size);
    if (im2col_status != kTfLiteOk) return im2col_status;
  }

  if (data->need_hwcn_weights) {
    node->temporaries->data[data->hwcn_weights_index] = data->hwcn_weights_id;
    TfLiteIntArray* hwcn_weights_size = TfLiteIntArrayCreate(2);

    // Because we're treating the filter weights as a matrix when we do the
    // transpose, we allocate the buffer with a two-dimensional shape, where one
    // dimension is the number of elements in each filter, and the second is the
    // total number of filters.
    int input_depth = input->dims->data[3];
    hwcn_weights_size->data[0] = (filter_height * filter_width * input_depth);
    hwcn_weights_size->data[1] = channels_out;

    TfLiteTensor* hwcn_weights =
        &context->tensors[node->temporaries->data[data->hwcn_weights_index]];
    hwcn_weights->type = input_type;
    hwcn_weights->allocation_type = kTfLiteArenaRwPersistent;

    auto hwcn_weights_status =
        context->ResizeTensor(context, hwcn_weights, hwcn_weights_size);
    if (hwcn_weights_status != kTfLiteOk) return hwcn_weights_status;

    // TODO(petewarden): If Resize() is called when the size hasn't actually
    // changed, this will do extra redundant work.
    data->have_weights_been_transposed = false;
  }

  if (is_hybrid) {
    node->temporaries->data[data->input_quantized_index] =
        data->input_quantized_id;
    TfLiteTensor* input_quantized;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, data->input_quantized_index,
                                  &input_quantized));
    input_quantized->type = kTfLiteInt8;
    input_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(input_quantized->dims, input->dims)) {
      TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                       input_quantized_size));
    }

    node->temporaries->data[data->scaling_factors_index] =
        data->scaling_factors_id;
    TfLiteTensor* scaling_factors;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, data->scaling_factors_index,
                                  &scaling_factors));
    scaling_factors->type = kTfLiteFloat32;
    scaling_factors->allocation_type = kTfLiteArenaRw;
    // Only one scale factor per batch is typically necessary. See optimized
    // implementation for why we need to allocate for the height of the inputs
    // flattened to 2D.
    TF_LITE_ENSURE(context, channels_in != 0);
    const int height = NumElements(input) / channels_in;
    int scaling_dims[1] = {height};
    if (!TfLiteIntArrayEqualsArray(scaling_factors->dims, 1, scaling_dims)) {
      TfLiteIntArray* scaling_factors_size = TfLiteIntArrayCreate(1);
      scaling_factors_size->data[0] = height;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scaling_factors,
                                                       scaling_factors_size));
    }

    node->temporaries->data[data->accum_scratch_index] = data->accum_scratch_id;
    TfLiteTensor* accum_scratch;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, data->accum_scratch_index,
                                       &accum_scratch));
    accum_scratch->type = kTfLiteInt32;
    accum_scratch->allocation_type = kTfLiteArenaRw;
    const int scratch_width = batches * out_height * out_width;
    int accum_scratch_dims[2] = {channels_out, scratch_width};
    if (!TfLiteIntArrayEqualsArray(accum_scratch->dims, 2,
                                   accum_scratch_dims)) {
      TfLiteIntArray* accum_scratch_size = TfLiteIntArrayCreate(2);
      accum_scratch_size->data[0] = channels_out;
      accum_scratch_size->data[1] = scratch_width;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, accum_scratch,
                                                       accum_scratch_size));
    }

    if (data->is_hybrid_per_channel) {
      const auto* affine_quantization =
          reinterpret_cast<TfLiteAffineQuantization*>(
              filter->quantization.params);
      TF_LITE_ENSURE_EQ(
          context, affine_quantization->scale->size,
          filter->dims->data[affine_quantization->quantized_dimension]);
      node->temporaries->data[data->input_offset_index] = data->input_offset_id;
      TfLiteTensor* input_offsets;
      TF_LITE_ENSURE_OK(
          context, GetTemporarySafe(context, node, data->input_offset_index,
                                    &input_offsets));
      input_offsets->type = kTfLiteInt32;
      input_offsets->allocation_type = kTfLiteArenaRw;
      // See above comment for the need to allocate for height of inputs.
      TF_LITE_ENSURE(context, channels_in != 0);
      const int height = NumElements(input) / channels_in;
      const int input_offset_dims[1] = {height};
      if (!TfLiteIntArrayEqualsArray(input_offsets->dims, 1,
                                     input_offset_dims)) {
        TfLiteIntArray* input_offsets_size = TfLiteIntArrayCreate(1);
        input_offsets_size->data[0] = input_offset_dims[0];
        TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_offsets,
                                                         input_offsets_size));
      }
      node->temporaries->data[data->row_sums_index] = data->row_sums_id;
      TfLiteTensor* row_sums;
      TF_LITE_ENSURE_OK(
          context,
          GetTemporarySafe(context, node, data->row_sums_index, &row_sums));
      row_sums->type = kTfLiteInt32;
      row_sums->allocation_type = kTfLiteArenaRwPersistent;
      // See above comment for the need to allocate for height of inputs.
      const int row_sums_dims[1] = {channels_out};
      if (!TfLiteIntArrayEqualsArray(row_sums->dims, 1, row_sums_dims)) {
        TfLiteIntArray* row_sums_size = TfLiteIntArrayCreate(1);
        row_sums_size->data[0] = row_sums_dims[0];
        TF_LITE_ENSURE_OK(
            context, context->ResizeTensor(context, row_sums, row_sums_size));
      }
    }
  }

  if (data->low_precision_applicable){
    int num_kernel_scratchpads = data->kernel_temps;
    int num_input_scratchpads  = data->input_temps ;
    int num_output_scratchpads = data->output_temps;

    // Allocating Filter and Required Kernel Scratchpads
    if (num_kernel_scratchpads >= 1){ // Filter Tensor
      int tensor_idx = data->filter_temps_idx;
      node->temporaries->data[tensor_idx] = data->filter_temps_id;
      TfLiteTensor* tensor = GetTemporary(context, node, /*index=*/tensor_idx);
      tensor->type = kTfLiteInt8;
      tensor->allocation_type = kTfLitePersistentRo;
      LowPrecision::Shape tensor_shape = ulp_params.kernel_scratchpads_shape_list.back();
      if (!TfLiteIntArrayEqualsArray(tensor->dims, 2, tensor_shape.size)) {
        std::cerr << "\tAllocating Filter Shape: " << LowPrecision::get_shape_string(tensor_shape);
        std::cerr.flush();
        TfLiteIntArray* tensor_size = TfLiteIntArrayCreate(2);
        tensor_size->data[0] = tensor_shape.size[0];
        tensor_size->data[1] = tensor_shape.size[1];
        TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, tensor, tensor_size));
        std::cerr << " DONE" << std::endl;
      }
    }
    for (int i = 1 ; i < num_kernel_scratchpads ; i++){ // Kernel Scratchpads Tensor
      int tensor_idx = data->kernel_temps_idx + i - 1;
      node->temporaries->data[tensor_idx] = data->kernel_temps_id[i];
      TfLiteTensor* tensor = GetTemporary(context, node, /*index=*/tensor_idx);
      tensor->type = kTfLiteInt8;
      tensor->allocation_type = kTfLiteArenaRw;
      LowPrecision::Shape tensor_shape = ulp_params.kernel_scratchpads_shape_list[num_kernel_scratchpads - i - 1];
      if (!TfLiteIntArrayEqualsArray(tensor->dims, 2, tensor_shape.size)) {
        std::cerr << "\tAllocating A Kernel Temporary Tensor With Shape: " << LowPrecision::get_shape_string(tensor_shape);
        std::cerr.flush();
        TfLiteIntArray* tensor_size = TfLiteIntArrayCreate(2);
        tensor_size->data[0] = tensor_shape.size[0];
        tensor_size->data[1] = tensor_shape.size[1];
        TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, tensor, tensor_size));
        std::cerr << " DONE" << std::endl;
      }
    }
    
    // Creating Filter Matrix
    TfLiteTensor* filter_tensor = GetTemporary(context, node, /*index=*/data->filter_temps_idx);
    int8_t* kernel_scratchpad_tensor = nullptr;
    if (num_kernel_scratchpads)
      kernel_scratchpad_tensor = LowPrecision::allocate<int8_t>(ulp_params.kernel_scratchpads_shape_list[num_kernel_scratchpads - 2].flatsize);
    data->filter_matrix = new LowPrecision::Matrix();
    data->filter_matrix->setDataAndPaddingAndScratchpadAndShape(
      GetTensorData<int8_t>(filter), 
      GetTensorData<int8_t>(filter_tensor), 
      kernel_scratchpad_tensor, 
      *ulp_params.filter_shape);
    if (num_kernel_scratchpads > 1)
        data->filter_matrix->setPaddingScratchpadSetting();
    data->filter_matrix->setNeedScratchpad();
    data->filter_matrix->setMemLayout(LowPrecision::MemLayout::kRowMajor);

    // Preparing Filter Matrix
    LowPrecision::Status filter_preparation_status;
    std::cerr << "\tPreparing Filter With Shape: " << LowPrecision::get_shape_string(*ulp_params.filter_shape);
    std::cerr.flush();
    filter_preparation_status = LowPrecision::PrepareMatrixAsFilterForMethod(*data->filter_matrix, ulp_params.method, data->timing_details);
    std::cerr << " DONE" << std::endl;
    TF_LITE_ASSERT_EQ(LowPrecision::mask_out_source(LowPrecision::report_on_failure(filter_preparation_status, data->low_precision_id, "CONV")), LowPrecision::Status::Success);
    if (num_kernel_scratchpads)
      LowPrecision::deallocate(kernel_scratchpad_tensor, true);

    // Allocating Required Input Scratchpads
    for (int i = 0 ; i < num_input_scratchpads ; i++){ // Input Scratchpads Tensor
      int tensor_idx = data->input_temps_idx + i;
      node->temporaries->data[tensor_idx] = data->input_temps_id[i];
      TfLiteTensor* tensor = GetTemporary(context, node, /*index=*/tensor_idx);
      tensor->type = kTfLiteInt8;
      tensor->allocation_type = kTfLiteArenaRw;
      LowPrecision::Shape tensor_shape = ulp_params.input_scratchpads_shape_list[num_input_scratchpads - 1 - i];
      if (!TfLiteIntArrayEqualsArray(tensor->dims, 2, tensor_shape.size)) {
        std::cerr << "\tAllocating An Input Temporary Tensor With Shape: " << LowPrecision::get_shape_string(tensor_shape);
        std::cerr.flush();
        TfLiteIntArray* tensor_size = TfLiteIntArrayCreate(2);
        tensor_size->data[0] = tensor_shape.size[0];
        tensor_size->data[1] = tensor_shape.size[1];
        TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, tensor, tensor_size));
        std::cerr << " DONE" << std::endl;
      }
    }
    
    // Allocating Required Output Scratchpads
    for (int i = 0 ; i < num_output_scratchpads ; i++){ // Output Scratchpads Tensor
      int tensor_idx = data->output_temps_idx + i;
      node->temporaries->data[tensor_idx] = data->output_temps_id[i];
      TfLiteTensor* tensor = GetTemporary(context, node, /*index=*/tensor_idx);
      tensor->type = kTfLiteInt32;
      tensor->allocation_type = kTfLiteArenaRw;
      LowPrecision::Shape tensor_shape = ulp_params.output_scratchpads_shape_list[num_output_scratchpads - 1 - i];
      if (!TfLiteIntArrayEqualsArray(tensor->dims, 2, tensor_shape.size)) {
        std::cerr << "\tAllocating An Output Temporary Tensor With Shape: " << LowPrecision::get_shape_string(tensor_shape);
        std::cerr.flush();
        TfLiteIntArray* tensor_size = TfLiteIntArrayCreate(2);
        tensor_size->data[0] = tensor_shape.size[0];
        tensor_size->data[1] = tensor_shape.size[1];
        TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, tensor, tensor_size));
        std::cerr << " DONE" << std::endl;
      }
    }
    
    /*std::cout << "\tAllocating LowPrecision Weight Tensors with Shape of ";
    LowPrecision::FullyConnected::set_default_method(ulp_params.method);

    node->temporaries->data[data->low_precision_weight_index] = data->low_precision_weight_id;

    TfLiteIntArray* filter_low_precision_quantization_size = TfLiteIntArrayCreate(2);

    int _filter_low_precision_quantization_size_a[] = { ulp_params.filter_shape->size[0], ulp_params.filter_shape->size[1] };
    LowPrecision::FullyConnected::TransformFilterShape(ulp_params.method, _filter_low_precision_quantization_size_a, 2);
    std::cout << "(" 
              << _filter_low_precision_quantization_size_a[0] 
              << ", "
              << _filter_low_precision_quantization_size_a[1] 
              << ")"
              << std::endl;

    filter_low_precision_quantization_size->data[0] = _filter_low_precision_quantization_size_a[0];
    filter_low_precision_quantization_size->data[1] = _filter_low_precision_quantization_size_a[1];

    TfLiteTensor* filter_low_precision_quantization =
        &context->tensors[node->temporaries->data[data->low_precision_weight_index]];
    filter_low_precision_quantization->type = kTfLiteInt8;
    filter_low_precision_quantization->allocation_type = kTfLitePersistentRo;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, filter_low_precision_quantization, filter_low_precision_quantization_size));
    LowPrecision::Status ret = LowPrecision::FullyConnected::QuantizeFilter(
      ulp_params.method, GetTensorData<int8_t>(filter),
      *(ulp_params.filter_shape), GetTensorData<int8_t>(filter_low_precision_quantization),
      LowPrecision::MemLayout::kRowMajor
    );
    TF_LITE_ASSERT_EQ(ret, LowPrecision::Status::Success);
    
    if (data->low_precision_activation_applicable || ulp_params.input_shape->size[0] > 1){
      std::cout << "\tAllocating LowPrecision Activations Tensors with Shape of ";
      node->temporaries->data[data->low_precision_activation_index] = data->low_precision_activation_id;
      TfLiteIntArray* activation_low_precision_quantization_size = TfLiteIntArrayCreate(2);

      int _activation_low_precision_quantization_size_a[] = { ulp_params.input_shape->size[0], ulp_params.input_shape->size[1] };
      LowPrecision::FullyConnected::TransformInputShape(ulp_params.method, _activation_low_precision_quantization_size_a, 2);
      std::cout << "(" 
                << _activation_low_precision_quantization_size_a[0] 
                << ", "
                << _activation_low_precision_quantization_size_a[1] 
                << ")"
                << std::endl;

      activation_low_precision_quantization_size->data[0] = _activation_low_precision_quantization_size_a[0];
      activation_low_precision_quantization_size->data[1] = _activation_low_precision_quantization_size_a[1];

      TfLiteTensor* activation_low_precision_quantization =
          &context->tensors[node->temporaries->data[data->low_precision_activation_index]];
      activation_low_precision_quantization->type = kTfLiteInt8;
      activation_low_precision_quantization->allocation_type = kTfLiteArenaRw;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, activation_low_precision_quantization, activation_low_precision_quantization_size));
    }*/
  }

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return Prepare(kernel_type, context, node);
}

template <KernelType kernel_type>
void EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                   TfLiteConvParams* params, OpData* data,
                   const TfLiteTensor* input, const TfLiteTensor* filter,
                   const TfLiteTensor* bias, TfLiteTensor* im2col,
                   TfLiteTensor* output) {
  auto input_offset = -input->params.zero_point;
  auto filter_offset = -filter->params.zero_point;
  auto output_offset = output->params.zero_point;

  KernelType effective_kernel_type;
  if ((kernel_type == kMultithreadOptimized ||
       kernel_type == kCblasOptimized) &&
      (params->dilation_width_factor != 1 ||
       params->dilation_height_factor != 1)) {
    // kMultithreadOptimized and kCblasOptimized do not support dilation.
    // Therefore, fallback to optimized.
    effective_kernel_type = kGenericOptimized;
  } else {
    effective_kernel_type = kernel_type;
  }

  // We have to fallback to reference execution path when im2col is needed but
  // disabled because to-be-allocated temporary im2col tensor is too large.
  // See b/178743262 for the detailed motivation.
  if (data->im2col_oversized) {
    effective_kernel_type = kReference;
  }

  ConvParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = -data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  switch (effective_kernel_type) {
    case kReference: {
      reference_ops::Conv(
          op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
          GetTensorShape(filter), GetTensorData<uint8_t>(filter),
          GetTensorShape(bias), GetTensorData<int32_t>(bias),
          GetTensorShape(output), GetTensorData<uint8_t>(output),
          GetTensorShape(im2col), GetTensorData<uint8_t>(im2col),
          /* cpu_backend_context = */ nullptr);
      break;
    }
    case kGenericOptimized:
    case kMultithreadOptimized:
    case kCblasOptimized: {
      // There is only one optimized implementation for Quantized Conv.
      optimized_ops::Conv(
          op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
          GetTensorShape(filter), GetTensorData<uint8_t>(filter),
          GetTensorShape(bias), GetTensorData<int32_t>(bias),
          GetTensorShape(output), GetTensorData<uint8_t>(output),
          GetTensorShape(im2col), GetTensorData<uint8_t>(im2col),
          CpuBackendContext::GetFromContext(context));
      break;
    }
  }
}

template <KernelType kernel_type>
void EvalQuantizedPerChannel(TfLiteContext* context, TfLiteNode* node,
                             TfLiteConvParams* params, OpData* data,
                             const TfLiteTensor* input,
                             const TfLiteTensor* filter,
                             const TfLiteTensor* bias, TfLiteTensor* output,
                             TfLiteTensor* im2col) {
  ConvParams op_params;
  op_params.input_offset = -input->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

  KernelType effective_kernel_type = kernel_type;
  // We have to fallback to reference execution path when im2col is needed but
  // disabled because to-be-allocated temporary im2col tensor is too large.
  // See b/178743262 for the detailed motivation.
  if (data->im2col_oversized) {
    effective_kernel_type = kReference;
  }
  if (data->low_precision_applicable && false){
    TfLiteTensor* filters = nullptr;
    TfLiteTensor* activations = nullptr;
    filters = GetTemporary(context, node, /*index=*/data->low_precision_weight_index);
    if (data->low_precision_activation_applicable){
      activations = GetTemporary(context, node, /*index=*/data->low_precision_activation_index);
    }
    // std::cout << "\tExecuting " << data->low_precision_id << std::endl;
    optimized_integer_ops::ConvPerChannel(
        op_params, data->per_channel_output_multiplier.data(),
        data->per_channel_output_shift.data(), 
        GetTensorShape(input),        GetTensorData<int8>(input), GetTensorData<int8>(activations),
        GetTensorShape(filter),       GetTensorData<int8>(filter), GetTensorData<int8>(filters),
        GetTensorShape(bias),         GetTensorData<int32>(bias),
        GetTensorShape(output),       GetTensorData<int8>(output),
        GetTensorShape(im2col),       GetTensorData<int8>(im2col),
        CpuBackendContext::GetFromContext(context));
  }
  else if (data->low_precision_applicable){
    int8_t* input_data  = nullptr;
    int8_t* output_data = GetTensorData<int8_t>(output);

    LowPrecision::Shape kernel_shape, input_shape, output_shape;

    optimized_integer_ops::ConvPerChannel(
        op_params, data->per_channel_output_multiplier.data(),
        data->per_channel_output_shift.data(), 
        GetTensorShape(input),        GetTensorData<int8>(input),
        GetTensorShape(filter),       GetTensorData<int8>(filter),
        GetTensorShape(bias),         GetTensorData<int32>(bias),
        GetTensorShape(output),       GetTensorData<int8>(output),
        GetTensorShape(im2col),       GetTensorData<int8>(im2col),
        CpuBackendContext::GetFromContext(context),
        &kernel_shape, &input_shape, &output_shape, &input_data);

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
    input_preparation_status = LowPrecision::PrepareMatrixAsInputForMethod(input_matrix, data->operation_method, data->timing_details);
    TF_LITE_ASSERT_EQ(LowPrecision::mask_out_source(LowPrecision::report_on_failure(input_preparation_status, data->low_precision_id, "CONV")), LowPrecision::Status::Success);
    
    // Getting Output Temporary Tensors
    std::vector<TfLiteTensor*> output_scratchpads(data->output_temps, nullptr);
    for (size_t i = 0; i < data->output_temps; i++)
      output_scratchpads[i] = GetTemporary(context, node, /*index=*/data->output_temps_idx + i);

    // Creating Output Matrix
    LowPrecision::Matrix output_matrix;
    output_matrix.setData(output_data);
    output_matrix.setShape(output_shape);
    if (data->output_temps >= 1){
      if (LowPrecision::FullyConnected::OutputPreProcess(data->operation_method) & LowPrecision::PreprocessType::Packing){
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
    output_preparation_status = LowPrecision::PrepareMatrixAsOutputForMethod(output_matrix, data->operation_method, data->timing_details);
    TF_LITE_ASSERT_EQ(LowPrecision::mask_out_source(LowPrecision::report_on_failure(output_preparation_status, data->low_precision_id, "CONV")), LowPrecision::Status::Success);

    LowPrecision::Status gemm_status;
    gemm_status = LowPrecision::GEMM(input_matrix, *data->filter_matrix, output_matrix, data->operation_method, data->timing_details);
    TF_LITE_ASSERT_EQ(LowPrecision::mask_out_source(LowPrecision::report_on_failure(gemm_status, data->low_precision_id, "CONV")), LowPrecision::Status::Success);
  }
  else
    switch (effective_kernel_type) {
      case kReference: {
        reference_integer_ops::ConvPerChannel(
            op_params, data->per_channel_output_multiplier.data(),
            data->per_channel_output_shift.data(), GetTensorShape(input),
            GetTensorData<int8>(input), GetTensorShape(filter),
            GetTensorData<int8>(filter), GetTensorShape(bias),
            GetTensorData<int32>(bias), GetTensorShape(output),
            GetTensorData<int8>(output));
        break;
      }
      case kGenericOptimized:
      case kMultithreadOptimized:
      case kCblasOptimized: {
        optimized_integer_ops::ConvPerChannel(
            op_params, data->per_channel_output_multiplier.data(),
            data->per_channel_output_shift.data(), GetTensorShape(input),
            GetTensorData<int8>(input), GetTensorShape(filter),
            GetTensorData<int8>(filter), GetTensorShape(bias),
            GetTensorData<int32>(bias), GetTensorShape(output),
            GetTensorData<int8>(output), GetTensorShape(im2col),
            GetTensorData<int8>(im2col),
            CpuBackendContext::GetFromContext(context));
        break;
      }
    }
}

template <KernelType kernel_type>
void EvalQuantizedPerChannel16x8(TfLiteContext* context, TfLiteNode* node,
                                 TfLiteConvParams* params, OpData* data,
                                 const TfLiteTensor* input,
                                 const TfLiteTensor* filter,
                                 const TfLiteTensor* bias, TfLiteTensor* output,
                                 TfLiteTensor* im2col) {
  ConvParams op_params;
  op_params.input_offset = -input->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

  switch (kernel_type) {
    case kGenericOptimized:
    case kMultithreadOptimized:
    case kCblasOptimized:
    case kReference: {
      reference_integer_ops::ConvPerChannel(
          op_params, data->per_channel_output_multiplier.data(),
          data->per_channel_output_shift.data(), GetTensorShape(input),
          GetTensorData<int16>(input), GetTensorShape(filter),
          GetTensorData<int8>(filter), GetTensorShape(bias),
          GetTensorData<std::int64_t>(bias), GetTensorShape(output),
          GetTensorData<int16>(output));
      break;
    }
  }
}

template <KernelType kernel_type>
void EvalFloat(TfLiteContext* context, TfLiteNode* node,
               TfLiteConvParams* params, OpData* data,
               const TfLiteTensor* input, const TfLiteTensor* filter,
               const TfLiteTensor* bias, TfLiteTensor* im2col,
               TfLiteTensor* hwcn_weights, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);
  KernelType effective_kernel_type = kernel_type;
  // Fall back to the optimized path if multi-threaded conv is unsupported.
  if ((kernel_type == kMultithreadOptimized) &&
      !data->supports_multithreaded_kernel) {
    effective_kernel_type = kGenericOptimized;
  }

  // When im2col is needed (which is implied when 'im2col_oversized' is true),
  // the GEMMM-based optimized path requires im2col data be allocated to ensure
  // the correctness. Therefore, when im2col is disabled because of the
  // oversized temporary im2col tensor, fallback to a non-optimized path is
  // needed.
  // See b/178743262 for the detailed motivation.
  if (data->im2col_oversized) {
    effective_kernel_type = kReference;
#if defined(TFLITE_WITH_MULTITHREADED_EIGEN)
    // As detailed by tflite::multithreaded_ops::Conv implementation in
    // multithreaded_conv.h, the Eigen-based execution doesn't need im2col data.
    // Therefore, we could rely on it as a better-optimized fallback than the
    // reference one.
    if (data->supports_multithreaded_kernel) {
      effective_kernel_type = kMultithreadOptimized;
    }
#endif
  }

  ConvParams op_params;
  op_params.padding_type = RuntimePaddingType(params->padding);
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  switch (effective_kernel_type) {
    case kReference: {
      reference_ops::Conv(op_params, GetTensorShape(input),
                          GetTensorData<float>(input), GetTensorShape(filter),
                          GetTensorData<float>(filter), GetTensorShape(bias),
                          GetTensorData<float>(bias), GetTensorShape(output),
                          GetTensorData<float>(output), GetTensorShape(im2col),
                          GetTensorData<float>(im2col));
      break;
    }
    case kCblasOptimized:
    case kGenericOptimized: {
      optimized_ops::Conv(op_params, GetTensorShape(input),
                          GetTensorData<float>(input), GetTensorShape(filter),
                          GetTensorData<float>(filter), GetTensorShape(bias),
                          GetTensorData<float>(bias), GetTensorShape(output),
                          GetTensorData<float>(output), GetTensorShape(im2col),
                          GetTensorData<float>(im2col),
                          CpuBackendContext::GetFromContext(context));
      break;
    }
    case kMultithreadOptimized: {
#if defined(TFLITE_WITH_MULTITHREADED_EIGEN)
      const float* filter_data;
      if (data->need_hwcn_weights) {
        filter_data = GetTensorData<float>(hwcn_weights);
      } else {
        filter_data = GetTensorData<float>(filter);
      }
      multithreaded_ops::Conv(
          *eigen_support::GetThreadPoolDevice(context), op_params,
          GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(filter), filter_data, GetTensorShape(bias),
          GetTensorData<float>(bias), GetTensorShape(output),
          GetTensorData<float>(output), GetTensorShape(im2col),
          GetTensorData<float>(im2col));
      break;
#else   // !defined(TFLITE_WITH_MULTITHREADED_EIGEN)
      // See Register_CONV_2D: we should never be here when TFLITE_WITH_RUY
      // was enabled. We #if out this code in order to get the corresponding
      // binary size benefits.
      TFLITE_DCHECK(false);
#endif  // defined(TFLITE_WITH_MULTITHREADED_EIGEN)
    }
  }
}

template <KernelType kernel_type>
TfLiteStatus EvalHybridPerChannel(TfLiteContext* context, TfLiteNode* node,
                                  TfLiteConvParams* params, OpData* data,
                                  const TfLiteTensor* input,
                                  const TfLiteTensor* filter,
                                  const TfLiteTensor* bias,
                                  TfLiteTensor* im2col, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  const int batch_size = SizeOfDimension(input, 0);
  TF_LITE_ENSURE(context, batch_size != 0);
  const int input_size = NumElements(input) / batch_size;
  TfLiteTensor* quantized_input_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, data->input_quantized_index,
                                     &quantized_input_tensor));
  int8_t* quantized_input_ptr_batch =
      GetTensorData<int8_t>(quantized_input_tensor);
  TfLiteTensor* scaling_factors_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, data->scaling_factors_index,
                                     &scaling_factors_tensor));
  float* scaling_factors_ptr = GetTensorData<float>(scaling_factors_tensor);
  TfLiteTensor* input_offset_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, data->input_offset_index,
                                     &input_offset_tensor));
  int32_t* input_offset_ptr = GetTensorData<int32_t>(input_offset_tensor);

  for (int b = 0; b < batch_size; ++b) {
    const int offset = b * input_size;
    tensor_utils::AsymmetricQuantizeFloats(
        GetTensorData<float>(input) + offset, input_size,
        quantized_input_ptr_batch + offset, &scaling_factors_ptr[b],
        &input_offset_ptr[b]);
  }

  int8_t* im2col_ptr = nullptr;
  int8_t* filter_ptr = nullptr;
  if (im2col != nullptr) {
    im2col_ptr = im2col->data.int8;
  }
  filter_ptr = filter->data.int8;
  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);

  KernelType effective_kernel_type = kernel_type;
  // We have to fallback to reference execution path when im2col is needed but
  // disabled because to-be-allocated temporary im2col tensor is too large.
  // See b/178743262 for the detailed motivation.
  if (data->im2col_oversized) {
    effective_kernel_type = kReference;
  }

  ConvParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  switch (effective_kernel_type) {
    case kReference:
      reference_ops::HybridConvPerChannel(
          op_params, scaling_factors_ptr, GetTensorShape(input),
          quantized_input_ptr_batch, GetTensorShape(filter), filter_ptr,
          GetTensorShape(bias), GetTensorData<float>(bias),
          GetTensorShape(output), GetTensorData<float>(output),
          GetTensorShape(im2col), im2col_ptr, affine_quantization->scale->data,
          input_offset_ptr);
      break;
    case kGenericOptimized:
    case kMultithreadOptimized:
    case kCblasOptimized: {
      TfLiteTensor* row_sums;
      TF_LITE_ENSURE_OK(
          context,
          GetTemporarySafe(context, node, data->row_sums_index, &row_sums));
      TfLiteTensor* scratch;
      TF_LITE_ENSURE_OK(
          context,
          GetTemporarySafe(context, node, data->accum_scratch_index, &scratch));
      TfLiteTensor* filter_low_precision = nullptr;
      TfLiteTensor* activation_low_precision = nullptr;
      // low_precision_applicable
      // low_precision_activation_applicable
      // low_precision_weight_index
      // low_precision_activation_index
      // low_precision_weight_id
      // low_precision_activation_id
      if (data->low_precision_applicable){
        filter_low_precision = GetTemporary(context, node, /*index=*/data->low_precision_weight_index);
      }
      // if (data->low_precision_multibatched){
      //   activation_low_precision = GetTemporary(context, node, /*index=*/6);
      // }
      if (data->low_precision_activation_applicable){
        activation_low_precision = GetTemporary(context, node, /*index=*/data->low_precision_activation_index);
      }
      optimized_ops::HybridConvPerChannel(
          op_params, scaling_factors_ptr, GetTensorShape(input),
          quantized_input_ptr_batch, GetTensorShape(filter), filter_ptr,
          GetTensorShape(bias), GetTensorData<float>(bias),
          GetTensorShape(output), GetTensorData<float>(output),
          GetTensorShape(im2col), im2col_ptr, affine_quantization->scale->data,
          input_offset_ptr, GetTensorShape(scratch),
          GetTensorData<int32>(scratch), GetTensorData<int32_t>(row_sums),
          GetTensorData<int8_t>(filter_low_precision),
          GetTensorData<int8_t>(activation_low_precision),
          &data->compute_hybrid_row_sums, &data->low_precision_applicable,
          &data->low_precision_activation_applicable, 
          CpuBackendContext::GetFromContext(context));
      data->compute_hybrid_row_sums = false;
      break;
    }
  }

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalHybrid(TfLiteContext* context, TfLiteNode* node,
                        TfLiteConvParams* params, OpData* data,
                        const TfLiteTensor* input, const TfLiteTensor* filter,
                        const TfLiteTensor* bias, TfLiteTensor* im2col,
                        TfLiteTensor* accum_scratch, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  const int batch_size = SizeOfDimension(input, 0);
  TF_LITE_ENSURE(context, batch_size != 0);
  const int input_size = NumElements(input) / batch_size;

  const float* input_ptr = GetTensorData<float>(input);
  TfLiteTensor* quantized_input_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, data->input_quantized_index,
                                     &quantized_input_tensor));
  int8_t* quantized_input_ptr_batch =
      GetTensorData<int8_t>(quantized_input_tensor);
  TfLiteTensor* scaling_factors_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, data->scaling_factors_index,
                                     &scaling_factors_tensor));
  float* scaling_factors_ptr = GetTensorData<float>(scaling_factors_tensor);

  // Per-batch input quantization for higher accuracy.
  {
    ruy::profiler::ScopeLabel label("ConvHybridQuantizeInputs");
    for (int b = 0; b < batch_size; ++b) {
      float unused_min, unused_max;
      const int offset = b * input_size;
      tensor_utils::SymmetricQuantizeFloats(
          input_ptr + offset, input_size, quantized_input_ptr_batch + offset,
          &unused_min, &unused_max, &scaling_factors_ptr[b]);
      scaling_factors_ptr[b] *= filter->params.scale;
    }
  }

  switch (kernel_type) {
    case kReference:
    case kGenericOptimized:
    case kMultithreadOptimized:
    case kCblasOptimized: {
      // There is only one implementation for hybrid kernel.
      ConvParams op_params;
      op_params.padding_type = PaddingType::kSame;
      op_params.padding_values.width = data->padding.width;
      op_params.padding_values.height = data->padding.height;
      op_params.stride_width = params->stride_width;
      op_params.stride_height = params->stride_height;
      op_params.dilation_width_factor = params->dilation_width_factor;
      op_params.dilation_height_factor = params->dilation_height_factor;
      op_params.float_activation_min = output_activation_min;
      op_params.float_activation_max = output_activation_max;
      optimized_ops::HybridConv(
          op_params, scaling_factors_ptr, GetTensorShape(input),
          quantized_input_ptr_batch, GetTensorShape(filter),
          GetTensorData<int8_t>(filter), GetTensorShape(bias),
          GetTensorData<float>(bias), GetTensorShape(accum_scratch),
          GetTensorData<int32_t>(accum_scratch), GetTensorShape(output),
          GetTensorData<float>(output), GetTensorShape(im2col),
          GetTensorData<int8_t>(im2col),
          CpuBackendContext::GetFromContext(context));
      break;
    }
  }

  return kTfLiteOk;
}

template <KernelType kernel_type, TfLiteType input_type>
TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  const TfLiteTensor* filter;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &filter));
  bool has_bias = node->inputs->size == 3;
  const TfLiteTensor* bias = has_bias ? GetInput(context, node, 2) : nullptr;
  TfLiteTensor* im2col =
      data->need_im2col
          ? &context->tensors[node->temporaries->data[data->im2col_index]]
          : nullptr;
  TfLiteTensor* hwcn_weights =
      data->need_hwcn_weights
          ? &context->tensors[node->temporaries->data[data->hwcn_weights_index]]
          : nullptr;

  if (data->need_hwcn_weights && !data->have_weights_been_transposed) {
    TransposeFloatTensor(filter, hwcn_weights);
    data->have_weights_been_transposed = true;
  }

  TFLITE_DCHECK_EQ(input_type, input->type);
  switch (input_type) {  // Already know in/outtypes are same.
    case kTfLiteFloat32:
      if (filter->type == kTfLiteUInt8 || filter->type == kTfLiteInt8) {
        if (data->is_hybrid_per_channel) {
          TF_LITE_ENSURE_OK(context, EvalHybridPerChannel<kernel_type>(
                                         context, node, params, data, input,
                                         filter, bias, im2col, output));
        } else {
          TfLiteTensor* accum_scratch =
              &context->tensors[node->temporaries
                                    ->data[data->accum_scratch_index]];
          TF_LITE_ENSURE_OK(context,
                            EvalHybrid<kernel_type>(context, node, params, data,
                                                    input, filter, bias, im2col,
                                                    accum_scratch, output));
        }
      } else {
        EvalFloat<kernel_type>(context, node, params, data, input, filter, bias,
                               im2col, hwcn_weights, output);
      }
      break;
    case kTfLiteUInt8:
      EvalQuantized<kernel_type>(context, node, params, data, input, filter,
                                 bias, im2col, output);
      break;
    case kTfLiteInt8:
      EvalQuantizedPerChannel<kernel_type>(context, node, params, data, input,
                                           filter, bias, output, im2col);
      break;
    case kTfLiteInt16:
      EvalQuantizedPerChannel16x8<kernel_type>(
          context, node, params, data, input, filter, bias, output, im2col);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s currently not supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));

  switch (input->type) {
    case kTfLiteFloat32:
      return EvalImpl<kernel_type, kTfLiteFloat32>(context, node);
    case kTfLiteUInt8:
      return EvalImpl<kernel_type, kTfLiteUInt8>(context, node);
    case kTfLiteInt8:
      return EvalImpl<kernel_type, kTfLiteInt8>(context, node);
    case kTfLiteInt16:
      return EvalImpl<kernel_type, kTfLiteInt16>(context, node);
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

}  // namespace conv

TfLiteRegistration* Register_CONVOLUTION_REF() {
  static TfLiteRegistration r = {conv::Init, conv::Free,
                                 conv::Prepare<conv::kReference>,
                                 conv::Eval<conv::kReference>};
  return &r;
}

TfLiteRegistration* Register_CONVOLUTION_GENERIC_OPT() {
  static TfLiteRegistration r = {conv::Init, conv::Free,
                                 conv::Prepare<conv::kGenericOptimized>,
                                 conv::Eval<conv::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_CONVOLUTION_GENERIC_OPT_UINT8() {
  static TfLiteRegistration r = {
      conv::Init, conv::Free, conv::Prepare<conv::kGenericOptimized>,
      conv::EvalImpl<conv::kGenericOptimized, kTfLiteUInt8>};
  return &r;
}

TfLiteRegistration* Register_CONVOLUTION_MULTITHREADED_OPT() {
  static TfLiteRegistration r = {conv::Init, conv::Free,
                                 conv::Prepare<conv::kMultithreadOptimized>,
                                 conv::Eval<conv::kMultithreadOptimized>};
  return &r;
}

TfLiteRegistration* Register_CONVOLUTION_CBLAS_OPT() {
  static TfLiteRegistration r = {conv::Init, conv::Free,
                                 conv::Prepare<conv::kCblasOptimized>,
                                 conv::Eval<conv::kCblasOptimized>};
  return &r;
}

TfLiteRegistration* Register_CONV_2D() {
#if defined TFLITE_USE_APPLE_ACCELERATE_FOR_CONV
  return Register_CONVOLUTION_CBLAS_OPT();
#elif defined TFLITE_WITH_MULTITHREADED_EIGEN
  return Register_CONVOLUTION_MULTITHREADED_OPT();
#else
  return Register_CONVOLUTION_GENERIC_OPT();
#endif
}

// Warning: Clients using this variant are responsible for ensuring that their
// models only need the UINT8 type. TFLite's op registration mechanism doesn't
// yet allow for more nuanced registration mechanisms.
TfLiteRegistration* Register_CONV_2D_UINT8() {
#if defined TFLITE_WITH_RUY
  // TFLITE_WITH_RUY optimizes the generic kernel type.
  return Register_CONVOLUTION_GENERIC_OPT_UINT8();
#else
  return Register_CONV_2D();
#endif
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

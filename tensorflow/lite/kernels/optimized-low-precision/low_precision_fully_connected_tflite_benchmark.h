#ifndef LOW_PRECISION_FULLY_CONNECTED_TFLITE_BENCHMARK_H_

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

#include <iostream>
#include <assert.h>

#include "common/types.h"

using namespace std;
using namespace LowPrecision;

void run_tflite_benchmark();

#endif
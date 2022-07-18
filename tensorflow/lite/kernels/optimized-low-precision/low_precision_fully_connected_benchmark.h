#include "low_precision_fully_connected.h"
#include "ruy/ruy.h"
#include "ruy/context.h"
#include <assert.h>
#include <stdio.h>
#include <fstream>
#include <string>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <tuple>
#include <math.h>

using namespace std;
using namespace LowPrecision;
using namespace LowPrecision::FullyConnected;

vector<pair<size_t, size_t>> extractSizesSingleBatch(std::string str);
vector<tuple<size_t, size_t, size_t>> extractSizesMultiBatch(std::string str);

typedef struct {
    bool        multibatch_benchmark = true;
    bool        singlebatch_benchmark = true;
    int         selected_benchmark_mode = 0xffffffff;

    bool        real_mul_api_benchmark_enable = true;
    int         real_mul_api_benchmark_mode = 0xffffffff;

    bool        real_single_mul_api_benchmark_enable = true;
    int         real_single_mul_api_benchmark_mode = 0xffffffff;

    bool        real_multi_mul_api_benchmark_enable = true;
    int         real_multi_mul_api_benchmark_mode = 0xffffffff;

    bool        single_mul_api_increasing_size_benchmark_enable = true;
    int         single_mul_api_increasing_size_benchmark_mode = 0xffffffff;

    bool        single_mul_api_different_size_benchmark_enable = true;
    int         single_mul_api_different_size_benchmark_mode = 0xffffffff;
    std::string single_mul_api_different_size_benchmark_time_path = "";
    std::string single_mul_api_different_size_benchmark_speedup_path = "";

    bool        multi_mul_api_different_size_benchmark_enable = true;
    int         multi_mul_api_different_size_benchmark_mode = 0xffffffff;
    std::string multi_mul_api_different_size_benchmark_time_path = "";
    std::string multi_mul_api_different_size_benchmark_speedup_path = "";
} benchmark_mode_t; 

double run_real_ruy_benchmark(int benchmark_iterations, Shape input_shape, Shape kernel_shape, Shape output_shape, bool disable_print = false, bool fill = false){
    if (!disable_print)
        cout << "\r[Int8Int8] Preparing";
    cout.flush();

    vector<int8_t*> input_vec       (benchmark_iterations, nullptr);
    vector<int8_t*> activation_vec  (benchmark_iterations, nullptr);
    vector<int32_t*>output_vec      (benchmark_iterations, nullptr);

    int8_t*  all_input_ptr       = allocate<int8_t> (input_shape.flatsize      * benchmark_iterations);
    int32_t* all_output_ptr      = allocate<int32_t>(output_shape.flatsize     * benchmark_iterations);
    int8_t*  filter_ptr          = allocate<int8_t> (kernel_shape.flatsize);

    if (fill){
        LowPrecision::one_minus_one_vector(all_input_ptr,   input_shape.flatsize  * benchmark_iterations);
        LowPrecision::one_minus_one_vector(filter_ptr,      kernel_shape.flatsize);
    }

    if (!disable_print)
        cout << "\r[Int8Int8] Setting Pointers";
    cout.flush();

    for (int i = 0 ; i < benchmark_iterations ; i++){
        input_vec.at(i)         = all_input_ptr      + i * input_shape.flatsize;
        output_vec.at(i)        = all_output_ptr     + i * output_shape.flatsize;
    }

    // Creating Context and Parameters
    ruy::Context* _ruy_context = new ruy::Context;
    ruy::MulParams<int32_t, int32_t> ruy_mul_params;
#ifdef DISABLE_KERNELS_MEM_ACCESS
    ruy::Matrix<int8_t> ruy_lhs;
    ruy::Matrix<int8_t> ruy_rhs;
    ruy::Matrix<int32_t> ruy_dst;

    // Creating Filter Matrix
    ruy::MakeSimpleLayout(
        kernel_shape.size[0], 
        kernel_shape.size[1], 
        ruy::Order::kRowMajor,
        ruy_lhs.mutable_layout()
    );
    ruy_lhs.set_data(filter_ptr);
    ruy_lhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);

    // Creating Output Matrix
    ruy::MakeSimpleLayout(
        (output_shape.number_dims == 2)?(output_shape.size[0]):(output_shape.size[0]), 
        (output_shape.number_dims == 2)?(output_shape.size[1]):(1),
        ruy::Order::kColMajor,
        ruy_dst.mutable_layout()
    );
    ruy_dst.set_data(output_vec.at(0));

    // Creating Input Matrix
    ruy::MakeSimpleLayout(
        (input_shape.number_dims == 2)?(input_shape.size[0]):(input_shape.size[0]), 
        (input_shape.number_dims == 2)?(input_shape.size[1]):(1),
        ruy::Order::kColMajor,
        ruy_rhs.mutable_layout()
    );
    ruy_rhs.set_data(input_vec.at(0));
#endif
    struct timespec tstart={0,0},
                    tend={0,0};

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
    for (int i = 0 ; i < benchmark_iterations ; i++){
        // Show Progress
        if ((i == 0 || i % ((int)(benchmark_iterations / 100)) == 0) && !disable_print){
            cout << "\r[Int8Int8] Processing Real Mul API with [ " 
                 << (((float)i) / benchmark_iterations) * 100 
                 << "% ]               ";
            cout.flush();
        }

#ifndef DISABLE_KERNELS_MEM_ACCESS
        ruy::Matrix<int8_t> ruy_lhs;
        ruy::Matrix<int8_t> ruy_rhs;
        ruy::Matrix<int32_t> ruy_dst;

        // Creating Filter Matrix
        ruy::MakeSimpleLayout(
            kernel_shape.size[0], 
            kernel_shape.size[1], 
            ruy::Order::kRowMajor,
            ruy_lhs.mutable_layout()
        );
        ruy_lhs.set_data(filter_ptr);
        ruy_lhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);

        // Creating Output Matrix
        ruy::MakeSimpleLayout(
            (output_shape.number_dims == 2)?(output_shape.size[1]):(output_shape.size[0]), 
            (output_shape.number_dims == 2)?(output_shape.size[0]):(1),
            ruy::Order::kColMajor,
            ruy_dst.mutable_layout()
        );
        ruy_dst.set_data(output_vec.at(i));

        // Creating Input Matrix
        ruy::MakeSimpleLayout(
            (input_shape.number_dims == 2)?(input_shape.size[1]):(input_shape.size[0]), 
            (input_shape.number_dims == 2)?(input_shape.size[0]):(1),
            ruy::Order::kColMajor,
            ruy_rhs.mutable_layout()
        );
        ruy_rhs.set_data(input_vec.at(i));
#endif
#ifdef DISABLE_KERNELS_MEM_ACCESS
        cerr << "Iterations " << i << endl;
#endif
        ruy::Mul(ruy_lhs, ruy_rhs, ruy_mul_params, _ruy_context, &ruy_dst);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);

    if (!disable_print)
        cout << "\r[Int8Int8] Ruy Mul API Finished Clearing...                           ";
    cout.flush();
    
    LowPrecision::deallocate(all_input_ptr);
    LowPrecision::deallocate(all_output_ptr);
    LowPrecision::deallocate(filter_ptr);

    long double time_consumed = calculate_time_diff_seconds(tstart, tend);

    if (!disable_print)
        cout << "\r[Int8Int8] Ruy Mul API: " << time_consumed << " s.                           ";
    cout.flush();

    return time_consumed;
}
double run_real_mul_api_benchmark(int benchmark_iterations, Shape input_shape, Shape kernel_shape, Shape output_shape, Method method, bool disable_print = false, bool fill = false){
    if (!disable_print)
        cout << "\r"
             << "[" << LowPrecision::get_method_string(method) << "] "
             << "Preparing";
    cout.flush();

    vector<int8_t*> input_vec       (benchmark_iterations, nullptr);
    vector<int8_t*> activation_vec  (benchmark_iterations, nullptr);
    vector<int32_t*>output_vec      (benchmark_iterations, nullptr);

    Shape activation_shape, filter_shape;
    activation_shape            = input_shape;
    filter_shape                = kernel_shape;

    activation_shape.flatsize   = LowPrecision::FullyConnected::TransformInputShape (method, activation_shape.size, activation_shape.number_dims);
    filter_shape.flatsize       = LowPrecision::FullyConnected::TransformFilterShape(method, filter_shape.size    , filter_shape.number_dims    );

    int8_t*  all_input_ptr       = allocate<int8_t> (input_shape.flatsize      * benchmark_iterations);
    int8_t*  all_activation_ptr  = allocate<int8_t> (activation_shape.flatsize * benchmark_iterations);
    int32_t* all_output_ptr      = allocate<int32_t>(output_shape.flatsize     * benchmark_iterations);
    int8_t*  filter_ptr          = allocate<int8_t> (filter_shape.flatsize);

    if (fill){
        LowPrecision::one_minus_one_vector(all_input_ptr,       input_shape.flatsize       * benchmark_iterations);
        LowPrecision::one_minus_one_vector(all_activation_ptr,  activation_shape.flatsize  * benchmark_iterations);
        LowPrecision::one_minus_one_vector(filter_ptr,          filter_shape.flatsize);
    }

    if (!disable_print)
        cout << "\r"
             << "[" << LowPrecision::get_method_string(method) << "] "
             << "Setting Pointers";
    cout.flush();

    for (int i = 0 ; i < benchmark_iterations ; i++){
        input_vec.at(i)         = all_input_ptr      + i * input_shape.flatsize;
        activation_vec.at(i)    = all_activation_ptr + i * activation_shape.flatsize;
        output_vec.at(i)        = all_output_ptr     + i * output_shape.flatsize;
    }

    // Creating Filter Matrix
    LowPrecision::Matrix filter_matrix;
    filter_matrix.setDataAndScratchpadAndShape(nullptr, filter_ptr, kernel_shape);
    filter_matrix.setNeedScratchpad();
    filter_matrix.setScratchpadValid();
    filter_matrix.setMemLayout(LowPrecision::MemLayout::kRowMajor);
#ifdef DISABLE_KERNELS_MEM_ACCESS
    // Creating Input Matrix
    LowPrecision::Matrix input_matrix;
    input_matrix.setDataAndScratchpadAndShape(input_vec.at(0), activation_vec.at(0), input_shape);
    input_matrix.setNeedScratchpad();
    input_matrix.setScratchpadValid();
    input_matrix.setMemLayout(LowPrecision::MemLayout::kRowMajor);

    // Creating Output Matrix
    LowPrecision::Matrix output_matrix;
    output_matrix.setDataAndScratchpadAndShape(output_vec.at(0), nullptr, output_shape);
    output_matrix.setMemLayout(LowPrecision::MemLayout::kRowMajor);
#endif
    struct timespec tstart={0,0},
                    tend={0,0};
    long long int calc_size = input_shape.size[0];
    calc_size *= kernel_shape.flatsize;
    calc_size *= benchmark_iterations;
    long long int calc_size_limit = 512;
    calc_size_limit *= 1024;
    calc_size_limit *= 1024;
    calc_size_limit *= 100;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
    for (int i = 0 ; i < benchmark_iterations ; i++){
        // Show Progress
        if ((i == 0 || i % ((int)(benchmark_iterations / 100)) == 0) 
            && !disable_print 
            && calc_size > calc_size_limit
        ){
            cout << "\r"
                 << "[" << LowPrecision::get_method_string(method) << "] "
                 << "Processing Real Mul API with [ " 
                 << (((float)i) / benchmark_iterations) * 100 
                 << "% ]               ";
            cout.flush();
        }

#ifndef DISABLE_KERNELS_MEM_ACCESS
        // Creating Input Matrix
        LowPrecision::Matrix input_matrix;
        input_matrix.setDataAndScratchpadAndShape(input_vec.at(i), activation_vec.at(i), input_shape);
        input_matrix.setNeedScratchpad();
        // input_matrix.setScratchpadValid();
        input_matrix.setMemLayout(LowPrecision::MemLayout::kRowMajor);

        // Creating Output Matrix
        LowPrecision::Matrix output_matrix;
        output_matrix.setDataAndScratchpadAndShape(output_vec.at(i), nullptr, output_shape);
        output_matrix.setMemLayout(LowPrecision::MemLayout::kRowMajor);

#endif
        // Multiplication
        LowPrecision::Status mul_ret = LowPrecision::FullyConnected::Mul(input_matrix, filter_matrix, output_matrix, method);

        // Check The Return Status
        if (LowPrecision::mask_out_source(mul_ret) != LowPrecision::Status::Success)
            return -1;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);

    if (!disable_print)
        cout << "\r"
             << "[" << LowPrecision::get_method_string(method) << "] "
             << "Real Mul API Finished Clearing...                           ";
    cout.flush();
    
    LowPrecision::deallocate(all_input_ptr);
    LowPrecision::deallocate(all_activation_ptr);
    LowPrecision::deallocate(all_output_ptr);
    LowPrecision::deallocate(filter_ptr);

    long double time_consumed = LowPrecision::calculate_time_diff_seconds(tstart, tend);

    if (!disable_print)
        cout << "\r"
             << "[" << LowPrecision::get_method_string(method) << "] "
             << "Real Mul API: " << time_consumed << " s.                           ";
    cout.flush();

    return time_consumed;
}

double run_i8i8_benchmark(int benchmark_iterations, Shape input_shape, Shape kernel_shape, Shape output_shape, bool disable_print = false){
    if (!disable_print)
        cout << "\rPreparing Benchmark Of Int8 Single-Batched For " << benchmark_iterations << " Iterations...";
    cout.flush();

    Shape filter_shape;
    filter_shape = kernel_shape;
    
    int8_t* input       = allocate<int8_t>(input_shape.flatsize);
    int8_t* filter      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output     = allocate<int32_t>(output_shape.flatsize);
    
    one_minus_one_vector(input, input_shape.flatsize);
    one_minus_one_vector(filter, kernel_shape.flatsize);

    ruy::Matrix<int8_t> ruy_lhs;
    ruy::Matrix<int8_t> ruy_rhs;
    ruy::Matrix<int32_t> ruy_dst;

    ruy::Context* _ruy_context = new ruy::Context;
    ruy::MulParams<int32_t, int32_t> ruy_mul_params;

    if (!disable_print)
        cout << "\rStarting Benchmark Of Int8 Single-Batched For " << benchmark_iterations << " Iterations...                           ";
    cout.flush();

    struct timespec tstart={0,0},
                    tend={0,0};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
    for(int i = 0 ; i < benchmark_iterations ; i++){
        if ((i == 0 || i % ((int)(benchmark_iterations / 100)) == 0) && !disable_print){
            cout << "\rProcessing Benchmark Of Int8 Single-Batched: [ " << (((float)i) / benchmark_iterations) * 100 << "% ]                              ";
            cout.flush();
        }

        // Create rhs
        ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kColMajor,
                                ruy_rhs.mutable_layout());
        ruy_rhs.set_data(filter);
        ruy_rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
        // Create dst
        ruy::MakeSimpleLayout(1, output_shape.size[0], ruy::Order::kColMajor,
                                ruy_dst.mutable_layout());
        ruy_dst.set_data(output);
        // Create lhs
        ruy::MakeSimpleLayout(1, input_shape.size[0], ruy::Order::kColMajor,
                                ruy_lhs.mutable_layout());
        ruy_lhs.set_data(input);

        ruy::Mul(ruy_lhs, ruy_rhs, ruy_mul_params, _ruy_context, &ruy_dst);

        zero_vector(input, input_shape.flatsize);
        zero_vector(output, output_shape.flatsize);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
    long double time_consumed = calculate_time_diff_seconds(tstart, tend);
    if (!disable_print)
        cout << "\rruy::Mul: " << time_consumed << " s.                                                                          ";
    cout.flush();
    return time_consumed;
}
double run_i8i4_benchmark(int benchmark_iterations, Shape input_shape, Shape kernel_shape, Shape output_shape, bool disable_print = false){
    if (!disable_print)
        cout << "\rPreparing Benchmark Of Int4 For " << benchmark_iterations << " Iterations...";
    cout.flush();

    Shape filter_shape;
    filter_shape = kernel_shape;

    filter_shape.flatsize = Int4::TransformFilterShape(filter_shape.size, filter_shape.number_dims);
    
    int8_t* input       = allocate<int8_t>(input_shape.flatsize);
    int8_t* kernel      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t* filter      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output     = allocate<int32_t>(output_shape.flatsize);
    
    one_minus_one_vector(input, input_shape.flatsize);
    one_minus_one_vector(kernel, kernel_shape.flatsize);
    
    Status weight_quantization_status = Int4::QuantizeFilter(kernel, kernel_shape, filter, MemLayout::kRowMajor);

    assert(weight_quantization_status == Status::Success);

    if (!disable_print)
        cout << "\rStarting Benchmark Of Int4 For " << benchmark_iterations << " Iterations...                    ";
    cout.flush();

    struct timespec tstart={0,0},
                    tend={0,0};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
    for(int i = 0 ; i < benchmark_iterations ; i++){
        if ((i == 0 || i % ((int)(benchmark_iterations / 100)) == 0) && !disable_print){
            cout << "\rProcessing Benchmark Of Int8 Single-Batched: [ " << (((float)i) / benchmark_iterations) * 100 << "% ]                              ";
            cout.flush();
        }
        Status multiplication_status = Int4::MultiplyInt8(
            input, input_shape,
            filter, kernel_shape,
            output, output_shape
        ); 

        zero_vector(output, output_shape.flatsize);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
    long double time_consumed = calculate_time_diff_seconds(tstart, tend);
    if (!disable_print)
        cout << "\rInt4::MultiplyInt8: " << time_consumed << " s.                           ";
    cout.flush();
    return time_consumed;
}
double run_i8bin_benchmark(int benchmark_iterations, Shape input_shape, Shape kernel_shape, Shape output_shape, bool disable_print = false){
    if (!disable_print)
        cout << "\rPreparing Benchmark Of Binary For " << benchmark_iterations << " Iterations...";
    cout.flush();

    Shape filter_shape;
    filter_shape = kernel_shape;
    
    filter_shape.flatsize = Binary::TransformFilterShape(filter_shape.size, filter_shape.number_dims);

    int8_t* input       = allocate<int8_t>(input_shape.flatsize);
    int8_t* kernel      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t* filter      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output     = allocate<int32_t>(output_shape.flatsize);

    one_minus_one_vector(input, input_shape.flatsize);
    one_minus_one_vector(kernel, kernel_shape.flatsize);

    Status weight_quantization_status = Binary::QuantizeFilter(kernel, kernel_shape, filter, MemLayout::kRowMajor);

    assert(weight_quantization_status == Status::Success);

    if (!disable_print)
        cout << "\rStarting Benchmark Of Binary For " << benchmark_iterations << " Iterations...";
    cout.flush();

    struct timespec tstart={0,0},
                    tend={0,0};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
    for(int i = 0 ; i < benchmark_iterations ; i++){
        if ((i == 0 || i % ((int)(benchmark_iterations / 100)) == 0) && !disable_print){
            cout << "\rProcessing Benchmark Of Binary Single-Batched: [ " << (((float)i) / benchmark_iterations) * 100 << "% ]                              ";
            cout.flush();
        }
        Status multiplication_status = Binary::MultiplyInt8SingleBatch(
            input, input_shape,
            filter, kernel_shape,
            output, output_shape
        ); 

        zero_vector(output, output_shape.flatsize);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
    long double time_consumed = calculate_time_diff_seconds(tstart, tend);
    if (!disable_print)
        cout << "\rBinary::MultiplyInt8SingleBatch: " << time_consumed << " s.                           ";
    cout.flush();
    return time_consumed;
}
double run_i8ter_benchmark(int benchmark_iterations, Shape input_shape, Shape kernel_shape, Shape output_shape, bool disable_print = false){
    if (!disable_print)
        cout << "\rPreparing Benchmark Of Ternary For " << benchmark_iterations << " Iterations...";
    cout.flush();

    Shape filter_shape;
    filter_shape = kernel_shape;
    
    filter_shape.flatsize = Ternary::TransformFilterShape(filter_shape.size, filter_shape.number_dims);

    int8_t* input       = allocate<int8_t>(input_shape.flatsize);
    int8_t* kernel      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t* filter      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output     = allocate<int32_t>(output_shape.flatsize);

    one_minus_one_vector(input, input_shape.flatsize);
    one_minus_one_vector(kernel, kernel_shape.flatsize);

    Status weight_quantization_status = Ternary::QuantizeFilter(kernel, kernel_shape, filter, MemLayout::kRowMajor);

    assert(weight_quantization_status == Status::Success);

    if (!disable_print)
        cout << "\rStarting Benchmark Of Ternary For " << benchmark_iterations << " Iterations...";
    cout.flush();

    struct timespec tstart={0,0},
                    tend={0,0};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
    for(int i = 0 ; i < benchmark_iterations ; i++){
        if ((i == 0 || i % ((int)(benchmark_iterations / 100)) == 0) && !disable_print){
            cout << "\rProcessing Benchmark Of Ternary Single-Batched: [ " << (((float)i) / benchmark_iterations) * 100 << "% ]                              ";
            cout.flush();
        }
        Status multiplication_status = Ternary::MultiplyInt8SingleBatch(
            input, input_shape,
            filter, kernel_shape,
            output, output_shape
        ); 

        zero_vector(output, output_shape.flatsize);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
    long double time_consumed = calculate_time_diff_seconds(tstart, tend);
    if (!disable_print)
        cout << "\rTernary::MultiplyInt8SingleBatch: " << time_consumed << " s.                           ";
    cout.flush();
    return time_consumed;
}
double run_i8qua_benchmark(int benchmark_iterations, Shape input_shape, Shape kernel_shape, Shape output_shape, bool disable_print = false){
    if (!disable_print)
        cout << "\rPreparing Benchmark Of Quaternary For " << benchmark_iterations << " Iterations...";
    cout.flush();

    Shape filter_shape;
    filter_shape = kernel_shape;
    
    filter_shape.flatsize = Quaternary::TransformFilterShape(filter_shape.size, filter_shape.number_dims);

    int8_t* input       = allocate<int8_t>(input_shape.flatsize);
    int8_t* kernel      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t* filter      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output     = allocate<int32_t>(output_shape.flatsize);

    one_minus_one_vector(input, input_shape.flatsize);
    one_minus_one_vector(kernel, kernel_shape.flatsize);

    Status weight_quantization_status = Quaternary::QuantizeFilter(kernel, kernel_shape, filter, MemLayout::kRowMajor);

    assert(weight_quantization_status == Status::Success);

    if (!disable_print)
        cout << "\rStarting Benchmark Of Quaternary For " << benchmark_iterations << " Iterations...";
    cout.flush();

    struct timespec tstart={0,0},
                    tend={0,0};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
    for(int i = 0 ; i < benchmark_iterations ; i++){
        if ((i == 0 || i % ((int)(benchmark_iterations / 100)) == 0) && !disable_print){
            cout << "\rProcessing Benchmark Of Quaternary Single-Batched: [ " << (((float)i) / benchmark_iterations) * 100 << "% ]                              ";
            cout.flush();
        }
        Status multiplication_status = Quaternary::MultiplyInt8SingleBatch(
            input, input_shape,
            filter, kernel_shape,
            output, output_shape
        ); 

        zero_vector(output, output_shape.flatsize);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
    long double time_consumed = calculate_time_diff_seconds(tstart, tend);
    if (!disable_print)
        cout << "\rQuaternary::MultiplyInt8SingleBatch: " << time_consumed << " s.                           ";
    cout.flush();
    return time_consumed;
}
double run_i4i8_benchmark(int benchmark_iterations, Shape input_shape, Shape kernel_shape, Shape output_shape, bool disable_print = false){
    if (!disable_print)
        cout << "\rPreparing Benchmark Of Int4InputsInt8Weights For " << benchmark_iterations << " Iterations...";
    cout.flush();

    Shape activation_shape, filter_shape;
    filter_shape = kernel_shape;
    activation_shape = input_shape;
    
    activation_shape.flatsize = Int4InputsInt8Weights::TransformInputShape(activation_shape.size, activation_shape.number_dims);
    filter_shape.flatsize = Int4InputsInt8Weights::TransformFilterShape(filter_shape.size, filter_shape.number_dims);

    int8_t* input       = allocate<int8_t>(input_shape.flatsize);
    int8_t* activation  = allocate<int8_t>(activation_shape.flatsize);
    int8_t* kernel      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t* filter      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output     = allocate<int32_t>(output_shape.flatsize);

    one_minus_one_vector(input, input_shape.flatsize);
    one_minus_one_vector(kernel, kernel_shape.flatsize);

    Status weight_quantization_status = Int4InputsInt8Weights::QuantizeFilter(kernel, kernel_shape, filter, MemLayout::kRowMajor);

    assert(weight_quantization_status == Status::Success);

    if (!disable_print)
        cout << "\rStarting Benchmark Of Int4InputsInt8Weights For " << benchmark_iterations << " Iterations...";
    cout.flush();

    struct timespec tstart={0,0},
                    tend={0,0};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
    for(int i = 0 ; i < benchmark_iterations ; i++){
        if ((i == 0 || i % ((int)(benchmark_iterations / 100)) == 0) && !disable_print){
            cout << "\rProcessing Benchmark Of Int4InputsInt8Weights Single-Batched: [ " << (((float)i) / benchmark_iterations) * 100 << "% ]                              ";
            cout.flush();
        }
        Status input_quantization_status = Int4InputsInt8Weights::QuantizeInput(input, input_shape, activation, MemLayout::kRowMajor); 

        Status multiplication_status = Int4InputsInt8Weights::MultiplyInt8SingleBatch(
            activation, input_shape,
            filter, kernel_shape,
            output, output_shape
        ); 

        zero_vector(activation, activation_shape.flatsize);
        zero_vector(output, output_shape.flatsize);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
    long double time_consumed = calculate_time_diff_seconds(tstart, tend);
    if (!disable_print)
        cout << "\rInt4InputsInt8Weights::MultiplyInt8SingleBatch: " << time_consumed << " s.                           ";
    cout.flush();
    return time_consumed;
}
double run_i4i4_benchmark(int benchmark_iterations, Shape input_shape, Shape kernel_shape, Shape output_shape, bool disable_print = false){
    if (!disable_print)
        cout << "\rPreparing Benchmark Of Int4InputsInt4Weights For " << benchmark_iterations << " Iterations...";
    cout.flush();

    Shape activation_shape, filter_shape;
    filter_shape = kernel_shape;
    activation_shape = input_shape;
    
    activation_shape.flatsize = Int4InputsInt4Weights::TransformInputShape(activation_shape.size, activation_shape.number_dims);
    filter_shape.flatsize = Int4InputsInt4Weights::TransformFilterShape(filter_shape.size, filter_shape.number_dims);

    int8_t* input       = allocate<int8_t>(input_shape.flatsize);
    int8_t* activation  = allocate<int8_t>(activation_shape.flatsize);
    int8_t* kernel      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t* filter      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output     = allocate<int32_t>(output_shape.flatsize);

    one_minus_one_vector(input, input_shape.flatsize);
    one_minus_one_vector(kernel, kernel_shape.flatsize);

    Status weight_quantization_status = Int4InputsInt4Weights::QuantizeFilter(kernel, kernel_shape, filter, MemLayout::kRowMajor);

    assert(weight_quantization_status == Status::Success);

    if (!disable_print)
        cout << "\rStarting Benchmark Of Int4InputsInt4Weights For " << benchmark_iterations << " Iterations...";
    cout.flush();

    struct timespec tstart={0,0},
                    tend={0,0};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
    for(int i = 0 ; i < benchmark_iterations ; i++){
        if ((i == 0 || i % ((int)(benchmark_iterations / 100)) == 0) && !disable_print){
            cout << "\rProcessing Benchmark Of Int4InputsInt4Weights Single-Batched: [ " << (((float)i) / benchmark_iterations) * 100 << "% ]                              ";
            cout.flush();
        }
        Status input_quantization_status = Int4InputsInt4Weights::QuantizeInput(input, input_shape, activation, MemLayout::kRowMajor); 

        Status multiplication_status = Int4InputsInt4Weights::MultiplyInt8SingleBatch(
            activation, input_shape,
            filter, kernel_shape,
            output, output_shape
        ); 

        zero_vector(activation, activation_shape.flatsize);
        zero_vector(output, output_shape.flatsize);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
    long double time_consumed = calculate_time_diff_seconds(tstart, tend);
    if (!disable_print)
        cout << "\rInt4InputsInt4Weights::MultiplyInt8SingleBatch: " << time_consumed << " s.                           ";
    cout.flush();
    return time_consumed;
}
double run_teri8_benchmark(int benchmark_iterations, Shape input_shape, Shape kernel_shape, Shape output_shape, bool disable_print = false){
    if (!disable_print)
        cout << "\rPreparing Benchmark Of TernaryInputsInt8Weights For " << benchmark_iterations << " Iterations...";
    cout.flush();

    Shape activation_shape, filter_shape;
    filter_shape = kernel_shape;
    activation_shape = input_shape;
    
    activation_shape.flatsize = TernaryInputsInt8Weights::TransformInputShape(activation_shape.size, activation_shape.number_dims);
    filter_shape.flatsize = TernaryInputsInt8Weights::TransformFilterShape(filter_shape.size, filter_shape.number_dims);

    int8_t* input       = allocate<int8_t>(input_shape.flatsize);
    int8_t* activation  = allocate<int8_t>(activation_shape.flatsize);
    int8_t* kernel      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t* filter      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output     = allocate<int32_t>(output_shape.flatsize);

    one_minus_one_vector(input, input_shape.flatsize);
    one_minus_one_vector(kernel, kernel_shape.flatsize);

    Status weight_quantization_status = TernaryInputsInt8Weights::QuantizeFilter(kernel, kernel_shape, filter, MemLayout::kRowMajor);

    assert(weight_quantization_status == Status::Success);

    if (!disable_print)
        cout << "\rStarting Benchmark Of TernaryInputsInt8Weights For " << benchmark_iterations << " Iterations...";
    cout.flush();

    struct timespec tstart={0,0},
                    tend={0,0};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
    for(int i = 0 ; i < benchmark_iterations ; i++){
        if ((i == 0 || i % ((int)(benchmark_iterations / 100)) == 0) && !disable_print){
            cout << "\rProcessing Benchmark Of TernaryInputsInt8Weights Single-Batched: [ " << (((float)i) / benchmark_iterations) * 100 << "% ]                              ";
            cout.flush();
        }
        Status input_quantization_status = TernaryInputsInt8Weights::QuantizeInput(input, input_shape, activation, MemLayout::kRowMajor); 

        Status multiplication_status = TernaryInputsInt8Weights::MultiplyInt8SingleBatch(
            activation, input_shape,
            filter, kernel_shape,
            output, output_shape
        ); 

        zero_vector(activation, activation_shape.flatsize);
        zero_vector(output, output_shape.flatsize);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
    long double time_consumed = calculate_time_diff_seconds(tstart, tend);
    if (!disable_print)
        cout << "\rTernaryInputsInt8Weights::MultiplyInt8SingleBatch: " << time_consumed << " s.                           ";
    cout.flush();
    return time_consumed;
}
double run_bini8_benchmark(int benchmark_iterations, Shape input_shape, Shape kernel_shape, Shape output_shape, bool disable_print = false){
    if (!disable_print)
        cout << "\rPreparing Benchmark Of BinaryInputsInt8Weights For " << benchmark_iterations << " Iterations...";
    cout.flush();

    Shape activation_shape, filter_shape;
    filter_shape = kernel_shape;
    activation_shape = input_shape;
    
    activation_shape.flatsize = BinaryInputsInt8Weights::TransformInputShape(activation_shape.size, activation_shape.number_dims);
    filter_shape.flatsize = BinaryInputsInt8Weights::TransformFilterShape(filter_shape.size, filter_shape.number_dims);

    int8_t* input       = allocate<int8_t>(input_shape.flatsize);
    int8_t* activation  = allocate<int8_t>(activation_shape.flatsize);
    int8_t* kernel      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t* filter      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output     = allocate<int32_t>(output_shape.flatsize);

    one_minus_one_vector(input, input_shape.flatsize);
    one_minus_one_vector(kernel, kernel_shape.flatsize);

    Status weight_quantization_status = BinaryInputsInt8Weights::QuantizeFilter(kernel, kernel_shape, filter, MemLayout::kRowMajor);

    assert(weight_quantization_status == Status::Success);

    if (!disable_print)
        cout << "\rStarting Benchmark Of BinaryInputsInt8Weights For " << benchmark_iterations << " Iterations...";
    cout.flush();

    struct timespec tstart={0,0},
                    tend={0,0};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
    for(int i = 0 ; i < benchmark_iterations ; i++){
        if ((i == 0 || i % ((int)(benchmark_iterations / 100)) == 0) && !disable_print){
            cout << "\rProcessing Benchmark Of BinaryInputsInt8Weights Single-Batched: [ " << (((float)i) / benchmark_iterations) * 100 << "% ]                              ";
            cout.flush();
        }
        Status input_quantization_status = BinaryInputsInt8Weights::QuantizeInput(input, input_shape, activation, MemLayout::kRowMajor); 

        Status multiplication_status = BinaryInputsInt8Weights::MultiplyInt8SingleBatch(
            activation, input_shape,
            filter, kernel_shape,
            output, output_shape
        ); 

        zero_vector(activation, activation_shape.flatsize);
        zero_vector(output, output_shape.flatsize);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
    long double time_consumed = calculate_time_diff_seconds(tstart, tend);
    if (!disable_print)
        cout << "\rBinaryInputsInt8Weights::MultiplyInt8SingleBatch: " << time_consumed << " s.                           ";
    cout.flush();
    return time_consumed;
}
double run_binbin_benchmark(int benchmark_iterations, Shape input_shape, Shape kernel_shape, Shape output_shape, bool disable_print = false){
    if (!disable_print)
        cout << "\rPreparing Benchmark Of BinaryInputsBinaryWeights For " << benchmark_iterations << " Iterations...";
    cout.flush();

    Shape activation_shape, filter_shape;
    filter_shape = kernel_shape;
    activation_shape = input_shape;
    
    activation_shape.flatsize = BinaryInputsBinaryWeights::TransformInputShape(activation_shape.size, activation_shape.number_dims);
    filter_shape.flatsize = BinaryInputsBinaryWeights::TransformFilterShape(filter_shape.size, filter_shape.number_dims);

    int8_t* input       = allocate<int8_t>(input_shape.flatsize);
    int8_t* activation  = allocate<int8_t>(activation_shape.flatsize);
    int8_t* kernel      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t* filter      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output     = allocate<int32_t>(output_shape.flatsize);

    one_minus_one_vector(input, input_shape.flatsize);
    one_minus_one_vector(kernel, kernel_shape.flatsize);

    Status weight_quantization_status = BinaryInputsBinaryWeights::QuantizeFilter(kernel, kernel_shape, filter, MemLayout::kRowMajor);

    assert(weight_quantization_status == Status::Success);

    if (!disable_print)
        cout << "\rStarting Benchmark Of BinaryInputsBinaryWeights For " << benchmark_iterations << " Iterations...";
    cout.flush();

    struct timespec tstart={0,0},
                    tend={0,0};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
    for(int i = 0 ; i < benchmark_iterations ; i++){
        if ((i == 0 || i % ((int)(benchmark_iterations / 100)) == 0) && !disable_print){
            cout << "\rProcessing Benchmark Of BinaryInputsBinaryWeights Single-Batched: [ " << (((float)i) / benchmark_iterations) * 100 << "% ]              ";
            cout.flush();
        }
        Status input_quantization_status = BinaryInputsBinaryWeights::QuantizeInput(input, input_shape, activation, MemLayout::kRowMajor); 

        Status multiplication_status = BinaryInputsBinaryWeights::MultiplyInt8SingleBatch(
            activation, input_shape,
            filter, kernel_shape,
            output, output_shape
        ); 

        zero_vector(activation, activation_shape.flatsize);
        zero_vector(output, output_shape.flatsize);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
    long double time_consumed = calculate_time_diff_seconds(tstart, tend);
    if (!disable_print)
        cout << "\rBinaryInputsBinaryWeights::MultiplyInt8SingleBatch: " << time_consumed << " s.                           ";
    cout.flush();
    return time_consumed;
}
double run_terter_benchmark(int benchmark_iterations, Shape input_shape, Shape kernel_shape, Shape output_shape, bool disable_print = false){
    if (!disable_print)
        cout << "\rPreparing Benchmark Of TernaryInputsTernaryWeights For " << benchmark_iterations << " Iterations...";
    cout.flush();

    Shape activation_shape, filter_shape;
    filter_shape = kernel_shape;
    activation_shape = input_shape;
    
    activation_shape.flatsize = TernaryInputsTernaryWeights::TransformInputShape(activation_shape.size, activation_shape.number_dims);
    filter_shape.flatsize = TernaryInputsTernaryWeights::TransformFilterShape(filter_shape.size, filter_shape.number_dims);

    int8_t* input       = allocate<int8_t>(input_shape.flatsize);
    int8_t* activation  = allocate<int8_t>(activation_shape.flatsize);
    int8_t* kernel      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t* filter      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output     = allocate<int32_t>(output_shape.flatsize);

    one_minus_one_vector(input, input_shape.flatsize);
    one_minus_one_vector(kernel, kernel_shape.flatsize);

    Status weight_quantization_status = TernaryInputsTernaryWeights::QuantizeFilter(kernel, kernel_shape, filter, MemLayout::kRowMajor);

    assert(weight_quantization_status == Status::Success);

    if (!disable_print)
        cout << "\rStarting Benchmark Of TernaryInputsTernaryWeights For " << benchmark_iterations << " Iterations...";
    cout.flush();

    struct timespec tstart={0,0},
                    tend={0,0};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
    for(int i = 0 ; i < benchmark_iterations ; i++){
        if ((i == 0 || i % ((int)(benchmark_iterations / 100)) == 0) && !disable_print){
            cout << "\rProcessing Benchmark Of TernaryInputsTernaryWeights Single-Batched: [ " << (((float)i) / benchmark_iterations) * 100 << "% ]                ";
            cout.flush();
        }
        Status input_quantization_status = TernaryInputsTernaryWeights::QuantizeInput(input, input_shape, activation, MemLayout::kRowMajor); 

        Status multiplication_status = TernaryInputsTernaryWeights::MultiplyInt8SingleBatch(
            activation, input_shape,
            filter, kernel_shape,
            output, output_shape
        ); 

        zero_vector(activation, activation_shape.flatsize);
        zero_vector(output, output_shape.flatsize);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
    long double time_consumed = calculate_time_diff_seconds(tstart, tend);
    if (!disable_print)
        cout << "\rTernaryInputsTernaryWeights::MultiplyInt8SingleBatch: " << time_consumed << " s.                           ";
    cout.flush();
    return time_consumed;}
double run_i3i3_benchmark(int benchmark_iterations, Shape input_shape, Shape kernel_shape, Shape output_shape, bool disable_print = false){
    if (!disable_print)
        cout << "\rPreparing Benchmark Of Int3InputsInt3Weights For " << benchmark_iterations << " Iterations...";
    cout.flush();

    Shape activation_shape, filter_shape;
    filter_shape = kernel_shape;
    activation_shape = input_shape;
    
    activation_shape.flatsize = Int3InputsInt3Weights::TransformInputShape(activation_shape.size, activation_shape.number_dims);
    filter_shape.flatsize = Int3InputsInt3Weights::TransformFilterShape(filter_shape.size, filter_shape.number_dims);

    int8_t* input       = allocate<int8_t>(input_shape.flatsize);
    int8_t* activation  = allocate<int8_t>(activation_shape.flatsize);
    int8_t* kernel      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t* filter      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output     = allocate<int32_t>(output_shape.flatsize);

    one_minus_one_vector(input, input_shape.flatsize);
    one_minus_one_vector(kernel, kernel_shape.flatsize);

    Status weight_quantization_status = Int3InputsInt3Weights::QuantizeFilter(kernel, kernel_shape, filter, MemLayout::kRowMajor);

    assert(weight_quantization_status == Status::Success);

    if (!disable_print)
        cout << "\rStarting Benchmark Of Int3InputsInt3Weights For " << benchmark_iterations << " Iterations...";
    cout.flush();

    struct timespec tstart={0,0},
                    tend={0,0};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
    for(int i = 0 ; i < benchmark_iterations ; i++){
        if ((i == 0 || i % ((int)(benchmark_iterations / 100)) == 0) && !disable_print){
            cout << "\rProcessing Benchmark Of Int3InputsInt3Weights Single-Batched: [ " << (((float)i) / benchmark_iterations) * 100 << "% ]                              ";
            cout.flush();
        }
        Status input_quantization_status = Int3InputsInt3Weights::QuantizeInput(input, input_shape, activation, MemLayout::kRowMajor); 

        Status multiplication_status = Int3InputsInt3Weights::MultiplyInt8SingleBatch(
            activation, input_shape,
            filter, kernel_shape,
            output, output_shape
        ); 

        zero_vector(activation, activation_shape.flatsize);
        zero_vector(output, output_shape.flatsize);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
    long double time_consumed = calculate_time_diff_seconds(tstart, tend);
    if (!disable_print)
        cout << "\rInt3InputsInt3Weights::MultiplyInt8SingleBatch: " << time_consumed << " s.                                                  ";
    cout.flush();
    return time_consumed;
}

double run_i8i8_mb_benchmark(int benchmark_iterations, Shape input_shape, Shape kernel_shape, Shape output_shape, bool disable_print = false){
    if (!disable_print)
        cout << "\rPreparing Benchmark Of Int8 Multi-Batched For " << benchmark_iterations << " Iterations...";
    cout.flush();

    Shape filter_shape;
    filter_shape = kernel_shape;
    
    int8_t* input       = allocate<int8_t>(input_shape.flatsize);
    int8_t* filter      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output     = allocate<int32_t>(output_shape.flatsize);
    
    one_minus_one_vector(input, input_shape.flatsize);
    one_minus_one_vector(filter, kernel_shape.flatsize);

    ruy::Matrix<int8_t> ruy_lhs;
    ruy::Matrix<int8_t> ruy_rhs;
    ruy::Matrix<int8_t> ruy_rhs_C;
    ruy::Matrix<int32_t> ruy_dst;

    ruy::Context* _ruy_context = new ruy::Context;
    ruy::MulParams<int32_t, int32_t> ruy_mul_params;

    if (!disable_print)
        cout << "\rStarting Benchmark Of Int8 Multi-Batched For " << benchmark_iterations << " Iterations...                           ";
    cout.flush();

    struct timespec tstart={0,0},
                    tend={0,0};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
    for(int i = 0 ; i < benchmark_iterations ; i++){
        if ((i == 0 || i % ((int)(benchmark_iterations / 100)) == 0) && !disable_print){
            cout << "\rProcessing Benchmark Of Int8 Multi-Batched: [ " << (((float)i) / benchmark_iterations) * 100 << "% ]                              ";
            cout.flush();
        }

        // Create rhs
        ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kColMajor,
                                ruy_rhs.mutable_layout());
        ruy_rhs.set_data(filter);
        ruy_rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
        // Create dst
        ruy::MakeSimpleLayout(output_shape.size[0], output_shape.size[1], ruy::Order::kColMajor,
                                ruy_dst.mutable_layout());
        ruy_dst.set_data(output);
        // Create lhs
        ruy::MakeSimpleLayout(input_shape.size[0], input_shape.size[1], ruy::Order::kColMajor,
                                ruy_lhs.mutable_layout());
        ruy_lhs.set_data(input);

        ruy::Mul(ruy_lhs, ruy_rhs, ruy_mul_params, _ruy_context, &ruy_dst);

        zero_vector(input, input_shape.flatsize);
        zero_vector(output, output_shape.flatsize);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
    long double time_consumed = calculate_time_diff_seconds(tstart, tend);
    if (!disable_print)
        cout << "\rruy::Mul: " << time_consumed << " s.                                                                          ";
    cout.flush();
    return time_consumed;
}
double run_i8i4_mb_benchmark(int benchmark_iterations, Shape input_shape, Shape kernel_shape, Shape output_shape, bool disable_print = false){
    if (!disable_print)
        cout << "\rPreparing Benchmark Of Int4 Multi-Batched For " << benchmark_iterations << " Iterations...";
    cout.flush();

    Shape filter_shape;
    filter_shape = kernel_shape;

    filter_shape.flatsize = Int4::TransformFilterShape(filter_shape.size, filter_shape.number_dims);
    
    int8_t* input       = allocate<int8_t>(input_shape.flatsize);
    int8_t* kernel      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t* filter      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output     = allocate<int32_t>(output_shape.flatsize);
    
    one_minus_one_vector(input, input_shape.flatsize);
    one_minus_one_vector(kernel, kernel_shape.flatsize);
    
    Status weight_quantization_status = Int4::QuantizeFilter(kernel, kernel_shape, filter, MemLayout::kRowMajor);

    assert(weight_quantization_status == Status::Success);

    if (!disable_print)
        cout << "\rStarting Benchmark Of Int4 Multi-Batched For " << benchmark_iterations << " Iterations...       ";
    cout.flush();

    struct timespec tstart={0,0},
                    tend={0,0};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
    for(int i = 0 ; i < benchmark_iterations ; i++){
        if ((i == 0 || i % ((int)(benchmark_iterations / 100)) == 0) && !disable_print){
            cout << "\rProcessing Benchmark Of Int4 Multi-Batched: [ " << (((float)i) / benchmark_iterations) * 100 << "% ]               ";
            cout.flush();
        }
        Status multiplication_status = Int4::MultiplyInt8MultiBatched(
            input, input_shape,
            filter, kernel_shape,
            output, output_shape
        ); 

        zero_vector(output, output_shape.flatsize);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
    long double time_consumed = calculate_time_diff_seconds(tstart, tend);
    if (!disable_print)
        cout << "\rInt4::MultiplyInt8MultiBatched: " << time_consumed << " s.                           ";
    cout.flush();
    return time_consumed;
}
double run_i8bin_mb_benchmark(int benchmark_iterations, Shape input_shape, Shape kernel_shape, Shape output_shape, bool disable_print = false){
    if (!disable_print)
        cout << "\rPreparing Benchmark Of Binary Multi-Batched For " << benchmark_iterations << " Iterations...";
    cout.flush();

    Shape filter_shape;
    filter_shape = kernel_shape;

    filter_shape.flatsize = Binary::TransformFilterShape(filter_shape.size, filter_shape.number_dims);
    
    int8_t* input       = allocate<int8_t>(input_shape.flatsize);
    int8_t* kernel      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t* filter      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output     = allocate<int32_t>(output_shape.flatsize);
    
    one_minus_one_vector(input, input_shape.flatsize);
    one_minus_one_vector(kernel, kernel_shape.flatsize);
    
    Status weight_quantization_status = Binary::QuantizeFilter(kernel, kernel_shape, filter, MemLayout::kRowMajor);

    assert(weight_quantization_status == Status::Success);

    if (!disable_print)
        cout << "\rStarting Benchmark Of Binary Multi-Batched For " << benchmark_iterations << " Iterations...       ";
    cout.flush();

    struct timespec tstart={0,0},
                    tend={0,0};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
    for(int i = 0 ; i < benchmark_iterations ; i++){
        if ((i == 0 || i % ((int)(benchmark_iterations / 100)) == 0) && !disable_print){
            cout << "\rProcessing Benchmark Of Binary Multi-Batched: [ " << (((float)i) / benchmark_iterations) * 100 << "% ]               ";
            cout.flush();
        }
        Status multiplication_status = Binary::MultiplyInt8MultiBatched(
            input, input_shape,
            filter, kernel_shape,
            output, output_shape
        ); 

        zero_vector(output, output_shape.flatsize);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
    long double time_consumed = calculate_time_diff_seconds(tstart, tend);
    if (!disable_print)
        cout << "\rBinary::MultiplyInt8MultiBatched: " << time_consumed << " s.                           ";
    cout.flush();
    return time_consumed;
}
double run_i8ter_mb_benchmark(int benchmark_iterations, Shape input_shape, Shape kernel_shape, Shape output_shape, bool disable_print = false){
    if (!disable_print)
        cout << "\rPreparing Benchmark Of Ternary Multi-Batched For " << benchmark_iterations << " Iterations...";
    cout.flush();

    Shape filter_shape;
    filter_shape = kernel_shape;

    filter_shape.flatsize = Ternary::TransformFilterShape(filter_shape.size, filter_shape.number_dims);
    
    int8_t* input       = allocate<int8_t>(input_shape.flatsize);
    int8_t* kernel      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t* filter      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output     = allocate<int32_t>(output_shape.flatsize);
    
    one_minus_one_vector(input, input_shape.flatsize);
    one_minus_one_vector(kernel, kernel_shape.flatsize);
    
    Status weight_quantization_status = Ternary::QuantizeFilter(kernel, kernel_shape, filter, MemLayout::kRowMajor);

    assert(weight_quantization_status == Status::Success);

    if (!disable_print)
        cout << "\rStarting Benchmark Of Ternary Multi-Batched For " << benchmark_iterations << " Iterations...       ";
    cout.flush();

    struct timespec tstart={0,0},
                    tend={0,0};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
    for(int i = 0 ; i < benchmark_iterations ; i++){
        if ((i == 0 || i % ((int)(benchmark_iterations / 100)) == 0) && !disable_print){
            cout << "\rProcessing Benchmark Of Ternary Multi-Batched: [ " << (((float)i) / benchmark_iterations) * 100 << "% ]               ";
            cout.flush();
        }
        Status multiplication_status = Ternary::MultiplyInt8MultiBatched(
            input, input_shape,
            filter, kernel_shape,
            output, output_shape
        ); 

        zero_vector(output, output_shape.flatsize);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
    long double time_consumed = calculate_time_diff_seconds(tstart, tend);
    if (!disable_print)
        cout << "\rTernary::MultiplyInt8MultiBatched: " << time_consumed << " s.                           ";
    cout.flush();
    return time_consumed;
}
double run_i4i8_mb_benchmark(int benchmark_iterations, Shape input_shape, Shape kernel_shape, Shape output_shape, bool disable_print = false){
    if (!disable_print)
        cout << "\rPreparing Benchmark Of Int4InputsInt8Weights Multi-Batched For " << benchmark_iterations << " Iterations...";
    cout.flush();

    Shape activation_shape, filter_shape;
    filter_shape = kernel_shape;
    activation_shape = input_shape;

    activation_shape.flatsize = Int4InputsInt8Weights::TransformInputShape(activation_shape.size, activation_shape.number_dims);
    filter_shape.flatsize = Int4InputsInt8Weights::TransformFilterShape(filter_shape.size, filter_shape.number_dims);
    
    int8_t* input       = allocate<int8_t>(input_shape.flatsize);
    int8_t* activation  = allocate<int8_t>(activation_shape.flatsize);
    int8_t* kernel      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t* filter      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output     = allocate<int32_t>(output_shape.flatsize);
    
    one_minus_one_vector(input, input_shape.flatsize);
    one_minus_one_vector(kernel, kernel_shape.flatsize);
    
    Status weight_quantization_status = Int4InputsInt8Weights::QuantizeFilter(kernel, kernel_shape, filter, MemLayout::kRowMajor);

    assert(weight_quantization_status == Status::Success);

    if (!disable_print)
        cout << "\rStarting Benchmark Of Int4InputsInt8Weights Multi-Batched For " << benchmark_iterations << " Iterations...       ";
    cout.flush();

    struct timespec tstart={0,0},
                    tend={0,0};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
    for(int i = 0 ; i < benchmark_iterations ; i++){
        if ((i == 0 || i % ((int)(benchmark_iterations / 100)) == 0) && !disable_print){
            cout << "\rProcessing Benchmark Of Int4InputsInt8Weights Multi-Batched: [ " << (((float)i) / benchmark_iterations) * 100 << "% ]               ";
            cout.flush();
        }
        Status input_quantization_status = Int4InputsInt8Weights::QuantizeInput(input, input_shape, activation, MemLayout::kRowMajor); 

        Status multiplication_status = Int4InputsInt8Weights::MultiplyInt8MultiBatched(
            input, input_shape,
            filter, kernel_shape,
            output, output_shape
        ); 

        zero_vector(activation, activation_shape.flatsize);
        zero_vector(output, output_shape.flatsize);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
    long double time_consumed = calculate_time_diff_seconds(tstart, tend);
    if (!disable_print)
        cout << "\rInt4InputsInt8Weights::MultiplyInt8MultiBatched: " << time_consumed << " s.                           ";
    cout.flush();
    return time_consumed;
}
double run_i4i4_mb_benchmark(int benchmark_iterations, Shape input_shape, Shape kernel_shape, Shape output_shape, bool disable_print = false){
    if (!disable_print)
        cout << "\rPreparing Benchmark Of Int4InputsInt4Weights Multi-Batched For " << benchmark_iterations << " Iterations...";
    cout.flush();

    Shape activation_shape, filter_shape;
    filter_shape = kernel_shape;
    activation_shape = input_shape;

    activation_shape.flatsize = Int4InputsInt4Weights::TransformInputShape(activation_shape.size, activation_shape.number_dims);
    filter_shape.flatsize = Int4InputsInt4Weights::TransformFilterShape(filter_shape.size, filter_shape.number_dims);
    
    int8_t* input       = allocate<int8_t>(input_shape.flatsize);
    int8_t* activation  = allocate<int8_t>(activation_shape.flatsize);
    int8_t* kernel      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t* filter      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output     = allocate<int32_t>(output_shape.flatsize);
    
    one_minus_one_vector(input, input_shape.flatsize);
    one_minus_one_vector(kernel, kernel_shape.flatsize);
    
    Status weight_quantization_status = Int4InputsInt4Weights::QuantizeFilter(kernel, kernel_shape, filter, MemLayout::kRowMajor);

    assert(weight_quantization_status == Status::Success);

    if (!disable_print)
        cout << "\rStarting Benchmark Of Int4InputsInt4Weights Multi-Batched For " << benchmark_iterations << " Iterations...       ";
    cout.flush();

    struct timespec tstart={0,0},
                    tend={0,0};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
    for(int i = 0 ; i < benchmark_iterations ; i++){
        if ((i == 0 || i % ((int)(benchmark_iterations / 100)) == 0) && !disable_print){
            cout << "\rProcessing Benchmark Of Int4InputsInt4Weights Multi-Batched: [ " << (((float)i) / benchmark_iterations) * 100 << "% ]               ";
            cout.flush();
        }
        Status input_quantization_status = Int4InputsInt4Weights::QuantizeInput(input, input_shape, activation, MemLayout::kRowMajor); 

        Status multiplication_status = Int4InputsInt4Weights::MultiplyInt8MultiBatched(
            input, input_shape,
            filter, kernel_shape,
            output, output_shape
        ); 

        zero_vector(activation, activation_shape.flatsize);
        zero_vector(output, output_shape.flatsize);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
    long double time_consumed = calculate_time_diff_seconds(tstart, tend);
    if (!disable_print)
        cout << "\rInt4InputsInt4Weights::MultiplyInt8MultiBatched: " << time_consumed << " s.                           ";
    cout.flush();
    return time_consumed;
}
double run_teri8_mb_benchmark(int benchmark_iterations, Shape input_shape, Shape kernel_shape, Shape output_shape, bool disable_print = false){
    if (!disable_print)
        cout << "\rPreparing Benchmark Of TernaryInputsInt8Weights Multi-Batched For " << benchmark_iterations << " Iterations...";
    cout.flush();

    Shape activation_shape, filter_shape;
    filter_shape = kernel_shape;
    activation_shape = input_shape;

    activation_shape.flatsize = TernaryInputsInt8Weights::TransformInputShape(activation_shape.size, activation_shape.number_dims);
    filter_shape.flatsize = TernaryInputsInt8Weights::TransformFilterShape(filter_shape.size, filter_shape.number_dims);
    
    int8_t* input       = allocate<int8_t>(input_shape.flatsize);
    int8_t* activation  = allocate<int8_t>(activation_shape.flatsize);
    int8_t* kernel      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t* filter      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output     = allocate<int32_t>(output_shape.flatsize);
    
    one_minus_one_vector(input, input_shape.flatsize);
    one_minus_one_vector(kernel, kernel_shape.flatsize);
    
    Status weight_quantization_status = TernaryInputsInt8Weights::QuantizeFilter(kernel, kernel_shape, filter, MemLayout::kRowMajor);

    assert(weight_quantization_status == Status::Success);

    if (!disable_print)
        cout << "\rStarting Benchmark Of TernaryInputsInt8Weights Multi-Batched For " << benchmark_iterations << " Iterations...       ";
    cout.flush();

    struct timespec tstart={0,0},
                    tend={0,0};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
    for(int i = 0 ; i < benchmark_iterations ; i++){
        if ((i == 0 || i % ((int)(benchmark_iterations / 100)) == 0) && !disable_print){
            cout << "\rProcessing Benchmark Of TernaryInputsInt8Weights Multi-Batched: [ " << (((float)i) / benchmark_iterations) * 100 << "% ]               ";
            cout.flush();
        }
        Status input_quantization_status = TernaryInputsInt8Weights::QuantizeInput(input, input_shape, activation, MemLayout::kRowMajor); 

        Status multiplication_status = TernaryInputsInt8Weights::MultiplyInt8MultiBatched(
            input, input_shape,
            filter, kernel_shape,
            output, output_shape
        ); 

        zero_vector(activation, activation_shape.flatsize);
        zero_vector(output, output_shape.flatsize);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
    long double time_consumed = calculate_time_diff_seconds(tstart, tend);
    if (!disable_print)
        cout << "\rTernaryInputsInt8Weights::MultiplyInt8MultiBatched: " << time_consumed << " s.                           ";
    cout.flush();
    return time_consumed;
}
double run_terter_mb_benchmark(int benchmark_iterations, Shape input_shape, Shape kernel_shape, Shape output_shape, bool disable_print = false){
    if (!disable_print)
        cout << "\rPreparing Benchmark Of TernaryInputsTernaryWeights Multi-Batched For " << benchmark_iterations << " Iterations...";
    cout.flush();

    Shape activation_shape, filter_shape;
    filter_shape = kernel_shape;
    activation_shape = input_shape;

    activation_shape.flatsize = TernaryInputsTernaryWeights::TransformInputShape(activation_shape.size, activation_shape.number_dims);
    filter_shape.flatsize = TernaryInputsTernaryWeights::TransformFilterShape(filter_shape.size, filter_shape.number_dims);
    
    int8_t* input       = allocate<int8_t>(input_shape.flatsize);
    int8_t* activation  = allocate<int8_t>(activation_shape.flatsize);
    int8_t* kernel      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t* filter      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output     = allocate<int32_t>(output_shape.flatsize);
    
    one_minus_one_vector(input, input_shape.flatsize);
    one_minus_one_vector(kernel, kernel_shape.flatsize);
    
    Status weight_quantization_status = TernaryInputsTernaryWeights::QuantizeFilter(kernel, kernel_shape, filter, MemLayout::kRowMajor);

    assert(weight_quantization_status == Status::Success);

    if (!disable_print)
        cout << "\rStarting Benchmark Of TernaryInputsTernaryWeights Multi-Batched For " << benchmark_iterations << " Iterations...       ";
    cout.flush();

    struct timespec tstart={0,0},
                    tend={0,0};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);
    for(int i = 0 ; i < benchmark_iterations ; i++){
        if ((i == 0 || i % ((int)(benchmark_iterations / 100)) == 0) && !disable_print){
            cout << "\rProcessing Benchmark Of TernaryInputsTernaryWeights Multi-Batched: [ " << (((float)i) / benchmark_iterations) * 100 << "% ]               ";
            cout.flush();
        }
        Status input_quantization_status = TernaryInputsTernaryWeights::QuantizeInput(input, input_shape, activation, MemLayout::kRowMajor); 

        Status multiplication_status = TernaryInputsTernaryWeights::MultiplyInt8MultiBatched(
            input, input_shape,
            filter, kernel_shape,
            output, output_shape
        ); 

        zero_vector(activation, activation_shape.flatsize);
        zero_vector(output, output_shape.flatsize);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
    long double time_consumed = calculate_time_diff_seconds(tstart, tend);
    if (!disable_print)
        cout << "\rTernaryInputsTernaryWeights::MultiplyInt8MultiBatched: " << time_consumed << " s.                           ";
    cout.flush();
    return time_consumed;
}

void run_benchmark(int benchmark_iterations, benchmark_mode_t benchmarks){
    int _num_batches          = 512,
        _num_inputs           = 512,
        _num_outputs          = 512;
    if (LowPrecision::FullyConnected::GetVariableFromEnv( "NumBatches" ) != "")
        _num_batches          = std::stoi(LowPrecision::FullyConnected::GetVariableFromEnv( "NumBatches" ));
    if (LowPrecision::FullyConnected::GetVariableFromEnv( "NumInputs" ) != "")
        _num_inputs           = std::stoi(LowPrecision::FullyConnected::GetVariableFromEnv( "NumInputs" ));
    if (LowPrecision::FullyConnected::GetVariableFromEnv( "NumOutputs" ) != "")
        _num_outputs          = std::stoi(LowPrecision::FullyConnected::GetVariableFromEnv( "NumOutputs" ));
    int num_batches           = _num_batches,
        num_inputs            = _num_inputs,
        num_outputs           = _num_outputs;
    int _input_shape[1]       = { num_inputs },
        _input_MB_shape[2]    = { num_batches, num_inputs },
        _kernel_shape[2]      = { num_outputs, num_inputs },
        _output_shape[1]      = { num_outputs },
        _output_MB_shape[2]   = { num_batches, num_outputs };
    Shape input_shape         = get_shape(_input_shape,      1),
          input_MB_shape      = get_shape(_input_MB_shape,   2),
          kernel_shape        = get_shape(_kernel_shape,     2),
          output_shape        = get_shape(_output_shape,     1),
          output_MB_shape     = get_shape(_output_MB_shape,  2);
    bool disable_progress = LowPrecision::FullyConnected::GetVariableFromEnv( "DisableProgress" ) == "TRUE";

    if (benchmarks.multibatch_benchmark && benchmarks.selected_benchmark_mode == 0xffff){
        double baseline_time = 0, benchmark_time;
        baseline_time  = run_i8i8_mb_benchmark(  benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape);
        cout << "\rBaseline Multibatch Execution Time : "
             << baseline_time << " seconds\n";
        benchmark_time = run_i8i4_mb_benchmark(  benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape);
        cout << "\r'i8i4' speedup : "
             << ((baseline_time - benchmark_time) / baseline_time) * 100 
             << "%                                                       \n";
        benchmark_time = run_i8bin_mb_benchmark( benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape);
        cout << "\r'i8bin' speedup : "
             << ((baseline_time - benchmark_time) / baseline_time) * 100 
             << "%                                                       \n";
        benchmark_time = run_i8ter_mb_benchmark( benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape);
        cout << "\r'i8ter' speedup : "
             << ((baseline_time - benchmark_time) / baseline_time) * 100 
             << "%                                                       \n";
        benchmark_time = run_i4i8_mb_benchmark(  benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape);
        cout << "\r'i4i8' speedup : "
             << ((baseline_time - benchmark_time) / baseline_time) * 100 
             << "%                                                       \n";
        benchmark_time = run_i4i4_mb_benchmark(  benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape);
        cout << "\r'i4i4' speedup : "
             << ((baseline_time - benchmark_time) / baseline_time) * 100 
             << "%                                                       \n";
        benchmark_time = run_teri8_mb_benchmark( benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape);
        cout << "\r'teri8' speedup : "
             << ((baseline_time - benchmark_time) / baseline_time) * 100 
             << "%                                                       \n";
        benchmark_time = run_terter_mb_benchmark(benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape);
        cout << "\r'terter' speedup : "
             << ((baseline_time - benchmark_time) / baseline_time) * 100 
             << "%                                                       \n";
    }
    else if (benchmarks.multibatch_benchmark){
        double benchmark_time = 0;
        if (benchmarks.selected_benchmark_mode & 0x8000){
            benchmark_time = run_i8i8_mb_benchmark(  benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape);
            cout << "\r'i8i8' Multibatch Execution Time : "      << benchmark_time << " seconds                                                       \n";
        }
        if (benchmarks.selected_benchmark_mode & 0x0001){
            benchmark_time = run_i8i4_mb_benchmark(  benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape);
            cout << "\r'i8i4' Multibatch Execution Time : "     << benchmark_time << " seconds                                                       \n";
        }
        if (benchmarks.selected_benchmark_mode & 0x0002){
            benchmark_time = run_i8bin_mb_benchmark( benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape);
            cout << "\r'i8bin' Multibatch Execution Time : "    << benchmark_time << " seconds                                                       \n";
        }
        if (benchmarks.selected_benchmark_mode & 0x0004){
            benchmark_time = run_i8ter_mb_benchmark( benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape);
            cout << "\r'i8ter' Multibatch Execution Time : "    << benchmark_time << " seconds                                                       \n";
        }
        if (benchmarks.selected_benchmark_mode & 0x0010){
            benchmark_time = run_i4i8_mb_benchmark(  benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape);
            cout << "\r'i4i8' Multibatch Execution Time : "     << benchmark_time << " seconds                                                       \n";
        }
        if (benchmarks.selected_benchmark_mode & 0x0020){
            benchmark_time = run_i4i4_mb_benchmark(  benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape);
            cout << "\r'i4i4' Multibatch Execution Time : "     << benchmark_time << " seconds                                                       \n";
        }
        if (benchmarks.selected_benchmark_mode & 0x0040){
            benchmark_time = run_teri8_mb_benchmark( benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape);
            cout << "\r'teri8' Multibatch Execution Time : "    << benchmark_time << " seconds                                                       \n";
        }
        if (benchmarks.selected_benchmark_mode & 0x0200){
            benchmark_time = run_terter_mb_benchmark(benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape);
            cout << "\r'terter' Multibatch Execution Time : "   << benchmark_time << " seconds                                                       \n";
    }}
    if (benchmarks.singlebatch_benchmark){
        double benchmark_time = 0;
        if (benchmarks.selected_benchmark_mode & 0x8000){
            benchmark_time = run_i8i8_benchmark(  benchmark_iterations, input_shape, kernel_shape, output_shape);
            cout << "\r'i8i8' Singlebatch Execution Time : "    << benchmark_time << " seconds                                                       \n";
        }
        if (benchmarks.selected_benchmark_mode & 0x0001){
            benchmark_time = run_i8i4_benchmark(  benchmark_iterations, input_shape, kernel_shape, output_shape);
            cout << "\r'i8i4' Singlebatch Execution Time : "    << benchmark_time << " seconds                                                       \n";
        }
        if (benchmarks.selected_benchmark_mode & 0x0002){
            benchmark_time = run_i8bin_benchmark( benchmark_iterations, input_shape, kernel_shape, output_shape);
            cout << "\r'i8bin' Singlebatch Execution Time : "   << benchmark_time << " seconds                                                       \n";
        }
        if (benchmarks.selected_benchmark_mode & 0x0004){
            benchmark_time = run_i8ter_benchmark( benchmark_iterations, input_shape, kernel_shape, output_shape);
            cout << "\r'i8ter' Singlebatch Execution Time : "   << benchmark_time << " seconds                                                       \n";
        }
        if (benchmarks.selected_benchmark_mode & 0x0008){
            benchmark_time = run_i8qua_benchmark( benchmark_iterations, input_shape, kernel_shape, output_shape);
            cout << "\r'i8qua' Singlebatch Execution Time : "   << benchmark_time << " seconds                                                       \n";
        }
        if (benchmarks.selected_benchmark_mode & 0x0010){
            benchmark_time = run_i4i8_benchmark(  benchmark_iterations, input_shape, kernel_shape, output_shape);
            cout << "\r'i4i8' Singlebatch Execution Time : "    << benchmark_time << " seconds                                                       \n";
        }
        if (benchmarks.selected_benchmark_mode & 0x0020){
            benchmark_time = run_i4i4_benchmark(  benchmark_iterations, input_shape, kernel_shape, output_shape);
            cout << "\r'i4i4' Singlebatch Execution Time : "    << benchmark_time << " seconds                                                       \n";
        }
        if (benchmarks.selected_benchmark_mode & 0x0080){
            benchmark_time = run_teri8_benchmark( benchmark_iterations, input_shape, kernel_shape, output_shape);
            cout << "\r'teri8' Singlebatch Execution Time : "   << benchmark_time << " seconds                                                       \n";
        }
        if (benchmarks.selected_benchmark_mode & 0x0100){
            benchmark_time = run_bini8_benchmark( benchmark_iterations, input_shape, kernel_shape, output_shape);
            cout << "\r'bini8' Singlebatch Execution Time : "   << benchmark_time << " seconds                                                       \n";
        }
        if (benchmarks.selected_benchmark_mode & 0x0040){
            benchmark_time = run_binbin_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape);
            cout << "\r'binbin' Singlebatch Execution Time : "  << benchmark_time << " seconds                                                       \n";
        }
        if (benchmarks.selected_benchmark_mode & 0x0200){
            benchmark_time = run_terter_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape);
            cout << "\r'terter' Singlebatch Execution Time : "  << benchmark_time << " seconds                                                       \n";
        }
        // if (benchmarks.selected_benchmark_mode & 0x0400){
        //     benchmark_time = run_i3i3_benchmark(  benchmark_iterations, input_shape, kernel_shape, output_shape);
        //     cout << "\r'i3i3' Singlebatch Execution Time : "    << benchmark_time << " seconds                                                       \n";
        // }
    }
    if (benchmarks.real_mul_api_benchmark_enable){
        cout << "Running Real Multi-Batch Mul API benchmark" << endl;
        bool show_speedups = LowPrecision::FullyConnected::GetVariableFromEnv( "ShowSpeedups" ) == "TRUE";
        double baseline_time = 1, benchmark_time;
        if (benchmarks.real_mul_api_benchmark_mode & 0x8000 || show_speedups){
            baseline_time = run_real_ruy_benchmark(benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape, disable_progress);
            if (show_speedups)
                cout << "\rBaseline Time: " << baseline_time << " seconds                   "  << endl;
            else
                cout << endl;
        }
        if (benchmarks.real_mul_api_benchmark_mode & 0x0001){
            benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape, LowPrecision::Method::kInt8Int4, disable_progress);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kInt8Int4) 
                     << "] time: " << benchmark_time 
                     << ", speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
        if (benchmarks.real_mul_api_benchmark_mode & 0x0002){
            benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape, LowPrecision::Method::kInt8Binary, disable_progress);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kInt8Binary) 
                     << "] time: " << benchmark_time 
                     << ", speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
        if (benchmarks.real_mul_api_benchmark_mode & 0x0004){
            benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape, LowPrecision::Method::kInt8Ternary, disable_progress);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kInt8Ternary) 
                     << "] time: " << benchmark_time 
                     << ", speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
        if (benchmarks.real_mul_api_benchmark_mode & 0x0010){
            benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape, LowPrecision::Method::kInt4ActInt8Weight, disable_progress);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kInt4ActInt8Weight) 
                     << "] time: " << benchmark_time 
                     << ", speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
        if (benchmarks.real_mul_api_benchmark_mode & 0x0020){
            benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape, LowPrecision::Method::kInt4ActInt4Weight, disable_progress);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kInt4ActInt4Weight) 
                     << "] time: " << benchmark_time 
                     << ", speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
        if (benchmarks.real_mul_api_benchmark_mode & 0x0040){
            benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape, LowPrecision::Method::kTernaryActInt8Weight, disable_progress);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kTernaryActInt8Weight) 
                     << "] time: " << benchmark_time 
                     << ", speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
        if (benchmarks.real_mul_api_benchmark_mode & 0x0200){
            benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape, LowPrecision::Method::kTernaryActTernaryWeight, disable_progress);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kTernaryActTernaryWeight) 
                     << "] time: " << benchmark_time 
                     << ", speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
        // if (benchmarks.real_mul_api_benchmark_mode & 0x0080){
        //    benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape, LowPrecision::Method::kBinaryActInt8Weight, disable_progress);
            // if (show_speedups)
            //     cout << "\r[" 
            //          << LowPrecision::get_method_string(LowPrecision::Method::kBinaryActInt8Weight) 
            //          << "] time: " << benchmark_time 
            //          << ", speedup: " 
            //          << (((baseline_time - benchmark_time) / baseline_time) * 100)
            //          << "%";
            // cout << endl;
        // }
        // if (benchmarks.real_mul_api_benchmark_mode & 0x0100){
        //    benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape, LowPrecision::Method::kBinaryActBinaryWeight, disable_progress);
            // if (show_speedups)
            //     cout << "\r[" 
            //          << LowPrecision::get_method_string(LowPrecision::Method::kBinaryActBinaryWeight) 
            //          << "] time: " << benchmark_time 
            //          << ", speedup: " 
            //          << (((baseline_time - benchmark_time) / baseline_time) * 100)
            //          << "%";
            // cout << endl;
        // }
        // if (benchmarks.real_mul_api_benchmark_mode & 0x0400){
        //    benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape, LowPrecision::Method::kInt3ActInt3Weight, disable_progress);
            // if (show_speedups)
            //     cout << "\r[" 
            //          << LowPrecision::get_method_string(LowPrecision::Method::kInt3ActInt3Weight) 
            //          << "] time: " << benchmark_time 
            //          << ", speedup: " 
            //          << (((baseline_time - benchmark_time) / baseline_time) * 100)
            //          << "%";
            // cout << endl;
        // }
        // if (benchmarks.real_mul_api_benchmark_mode & 0x0008){
        //     benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape, LowPrecision::Method::kInt8QuaTernary, disable_progress);
            // if (show_speedups)
            //     cout << "\r[" 
            //          << LowPrecision::get_method_string(LowPrecision::Method::kInt8QuaTernary) 
            //          << "] time: " << benchmark_time 
            //          << ", speedup: " 
            //          << (((baseline_time - benchmark_time) / baseline_time) * 100)
            //          << "%";
            // cout << endl;
        // }
    }
    if (benchmarks.real_single_mul_api_benchmark_enable){
        cout << "Running Real Single-Batch Mul API benchmark" << endl;
        bool show_speedups = LowPrecision::FullyConnected::GetVariableFromEnv( "ShowSpeedups" ) == "TRUE";
        double baseline_time = 1, benchmark_time;
        if (benchmarks.real_single_mul_api_benchmark_mode & 0x8000 || show_speedups){
            baseline_time = run_real_ruy_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape);
            if (show_speedups)
                cout << "\rBaseline Time: " << baseline_time << " seconds                   "  << endl;
            else
                cout << endl;
        }
        if (benchmarks.real_single_mul_api_benchmark_mode & 0x0001){
            benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape, LowPrecision::Method::kInt8Int4, show_speedups);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kInt8Int4) 
                     << "] speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
        if (benchmarks.real_single_mul_api_benchmark_mode & 0x0002){
            benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape, LowPrecision::Method::kInt8Binary, show_speedups);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kInt8Binary) 
                     << "] speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
        if (benchmarks.real_single_mul_api_benchmark_mode & 0x0004){
            benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape, LowPrecision::Method::kInt8Ternary, show_speedups);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kInt8Ternary) 
                     << "] speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
        if (benchmarks.real_single_mul_api_benchmark_mode & 0x0010){
            benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape, LowPrecision::Method::kInt4ActInt8Weight, show_speedups);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kInt4ActInt8Weight) 
                     << "] speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
        if (benchmarks.real_single_mul_api_benchmark_mode & 0x0020){
            benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape, LowPrecision::Method::kInt4ActInt4Weight, show_speedups);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kInt4ActInt4Weight) 
                     << "] speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
        if (benchmarks.real_single_mul_api_benchmark_mode & 0x0040){
            benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape, LowPrecision::Method::kTernaryActInt8Weight, show_speedups);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kTernaryActInt8Weight) 
                     << "] speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
        if (benchmarks.real_single_mul_api_benchmark_mode & 0x0200){
            benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape, LowPrecision::Method::kTernaryActTernaryWeight, show_speedups);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kTernaryActTernaryWeight) 
                     << "] speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
        if (benchmarks.real_single_mul_api_benchmark_mode & 0x0080){
           benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape, LowPrecision::Method::kBinaryActInt8Weight, show_speedups);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kBinaryActInt8Weight) 
                     << "] speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
        if (benchmarks.real_single_mul_api_benchmark_mode & 0x0100){
           benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape, LowPrecision::Method::kBinaryActBinaryWeight, show_speedups);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kBinaryActBinaryWeight) 
                     << "] speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
        // if (benchmarks.real_single_mul_api_benchmark_mode & 0x0400){
        //    benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape, LowPrecision::Method::kInt3ActInt3Weight, show_speedups);
        //     if (show_speedups)
        //         cout << "\r[" 
        //              << LowPrecision::get_method_string(LowPrecision::Method::kInt3ActInt3Weight) 
        //              << "] speedup: " 
        //              << (((baseline_time - benchmark_time) / baseline_time) * 100)
        //              << "%";
        //     cout << endl;
        // }
        if (benchmarks.real_single_mul_api_benchmark_mode & 0x0008){
            benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape, LowPrecision::Method::kInt8QuaTernary, show_speedups);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kInt8QuaTernary) 
                     << "] speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
    }
    if (benchmarks.real_multi_mul_api_benchmark_enable){
        cout << "Running Real Multi-Batch Mul API benchmark" << endl;
        bool show_speedups = LowPrecision::FullyConnected::GetVariableFromEnv( "ShowSpeedups" ) == "TRUE";
        double baseline_time = 1, benchmark_time;
        if (benchmarks.real_multi_mul_api_benchmark_mode & 0x8000 || show_speedups){
            baseline_time = run_real_ruy_benchmark(benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape);
            if (show_speedups)
                cout << "\rBaseline Time: " << baseline_time << " seconds                   "  << endl;
            else
                cout << endl;
        }
        if (benchmarks.real_multi_mul_api_benchmark_mode & 0x0001){
            benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape, LowPrecision::Method::kInt8Int4, disable_progress);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kInt8Int4) 
                     << "] speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
        if (benchmarks.real_multi_mul_api_benchmark_mode & 0x0002){
            benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape, LowPrecision::Method::kInt8Binary, disable_progress);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kInt8Binary) 
                     << "] speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
        if (benchmarks.real_multi_mul_api_benchmark_mode & 0x0004){
            benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape, LowPrecision::Method::kInt8Ternary, disable_progress);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kInt8Ternary) 
                     << "] speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
        if (benchmarks.real_multi_mul_api_benchmark_mode & 0x0010){
            benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape, LowPrecision::Method::kInt4ActInt8Weight, disable_progress);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kInt4ActInt8Weight) 
                     << "] speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
        if (benchmarks.real_multi_mul_api_benchmark_mode & 0x0020){
            benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape, LowPrecision::Method::kInt4ActInt4Weight, disable_progress);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kInt4ActInt4Weight) 
                     << "] speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
        if (benchmarks.real_multi_mul_api_benchmark_mode & 0x0040){
            benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape, LowPrecision::Method::kTernaryActInt8Weight, disable_progress);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kTernaryActInt8Weight) 
                     << "] speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
        if (benchmarks.real_multi_mul_api_benchmark_mode & 0x0200){
            benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_MB_shape, kernel_shape, output_MB_shape, LowPrecision::Method::kTernaryActTernaryWeight, disable_progress);
            if (show_speedups)
                cout << "\r[" 
                     << LowPrecision::get_method_string(LowPrecision::Method::kTernaryActTernaryWeight) 
                     << "] speedup: " 
                     << (((baseline_time - benchmark_time) / baseline_time) * 100)
                     << "%";
            cout << endl;
        }
    }
    if (benchmarks.single_mul_api_increasing_size_benchmark_enable){
        bool show_speedups = LowPrecision::FullyConnected::GetVariableFromEnv( "ShowSpeedups" ) == "TRUE";
        int  input_increase_coef = 4;
        int  output_increase_coef = 4;

        if (LowPrecision::FullyConnected::GetVariableFromEnv( "InputIncreaseCoef" ) != "")
            input_increase_coef  = std::stoi(LowPrecision::FullyConnected::GetVariableFromEnv( "InputIncreaseCoef" ));
        if (LowPrecision::FullyConnected::GetVariableFromEnv( "OutputIncreaseCoef" ) != "")
            output_increase_coef  = std::stoi(LowPrecision::FullyConnected::GetVariableFromEnv( "OutputIncreaseCoef" ));
        
        cout << "Running Single-Batch Mul API Increasing Size benchmark with input coefficient " 
             << input_increase_coef
             << " and output coefficient "
             << output_increase_coef
             << endl;

        int _input_shape_increased[1]  = { num_inputs  * input_increase_coef  },
            _kernel_shape_increased[2] = { num_outputs * output_increase_coef , num_inputs  * input_increase_coef },
            _output_shape_increased[1] = { num_outputs * output_increase_coef };
        Shape input_shape_increased    = get_shape(_input_shape_increased,       1),
              kernel_shape_increased   = get_shape(_kernel_shape_increased,      2),
              output_shape_increased   = get_shape(_output_shape_increased,      1);

        double baseline_before_time = 1, benchmark_before_time;
        double baseline_after_time = 1, benchmark_after_time;

        if (benchmarks.single_mul_api_increasing_size_benchmark_mode & 0x8000 || show_speedups){
            baseline_before_time = run_real_ruy_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape, disable_progress);
            baseline_after_time  = run_real_ruy_benchmark(benchmark_iterations, input_shape_increased, kernel_shape_increased, output_shape_increased, disable_progress);
            cout << "\rBaseline Time Increase: " 
                 << baseline_before_time << " -> " 
                 << baseline_after_time << " seconds ( " 
                 << (((baseline_after_time - baseline_before_time) / baseline_before_time) * 100) 
                 << "% increase )" << endl;
        }
        if (benchmarks.single_mul_api_increasing_size_benchmark_mode & 0x0001){
            benchmark_before_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape, LowPrecision::Method::kInt8Int4, disable_progress);
            benchmark_after_time  = run_real_mul_api_benchmark(benchmark_iterations, input_shape_increased, kernel_shape_increased, output_shape_increased, LowPrecision::Method::kInt8Int4, disable_progress);
            if (show_speedups){
                double speedup_before = 0, speedup_after = 0;
                speedup_before = ((baseline_before_time - benchmark_before_time) / baseline_before_time) * 100;
                speedup_after  = ((baseline_after_time  - benchmark_after_time)  / baseline_after_time)  * 100;
                cout << "\r[" 
                    << LowPrecision::get_method_string(LowPrecision::Method::kInt8Int4) 
                    << "] Speed-Up Increase: " << speedup_before << "% -> " << speedup_after << "% ( " 
                    << (((speedup_after - speedup_before) / abs(speedup_before)) * 100)
                    << "% increase )                ";
            }
            else{
                cout << "\r[" 
                    << LowPrecision::get_method_string(LowPrecision::Method::kInt8Int4) 
                    << "] Benchmark Time Increase: " << benchmark_before_time << " -> " << benchmark_after_time << " seconds ( " 
                    << (((benchmark_after_time - benchmark_before_time) / benchmark_before_time) * 100)
                    << "% increase )                ";
            }
            cout << endl;
        }
        if (benchmarks.single_mul_api_increasing_size_benchmark_mode & 0x0002){
            benchmark_before_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape, LowPrecision::Method::kInt8Binary, disable_progress);
            benchmark_after_time  = run_real_mul_api_benchmark(benchmark_iterations, input_shape_increased, kernel_shape_increased, output_shape_increased, LowPrecision::Method::kInt8Binary, disable_progress);
            if (show_speedups){
                double speedup_before = 0, speedup_after = 0;
                speedup_before = ((baseline_before_time - benchmark_before_time) / baseline_before_time) * 100;
                speedup_after  = ((baseline_after_time  - benchmark_after_time)  / baseline_after_time)  * 100;
                cout << "\r[" 
                    << LowPrecision::get_method_string(LowPrecision::Method::kInt8Binary) 
                    << "] Speed-Up Increase: " << speedup_before << "% -> " << speedup_after << "% ( " 
                    << (((speedup_after - speedup_before) / abs(speedup_before)) * 100)
                    << "% increase )                ";
            }
            else{
                cout << "\r[" 
                    << LowPrecision::get_method_string(LowPrecision::Method::kInt8Binary) 
                    << "] Benchmark Time Increase: " << benchmark_before_time << " -> " << benchmark_after_time << " seconds ( " 
                    << (((benchmark_after_time - benchmark_before_time) / benchmark_before_time) * 100)
                    << "% increase )                ";
            }
            cout << endl;
        }
        if (benchmarks.single_mul_api_increasing_size_benchmark_mode & 0x0004){
            benchmark_before_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape, LowPrecision::Method::kInt8Ternary, disable_progress);
            benchmark_after_time  = run_real_mul_api_benchmark(benchmark_iterations, input_shape_increased, kernel_shape_increased, output_shape_increased, LowPrecision::Method::kInt8Ternary, disable_progress);
            if (show_speedups){
                double speedup_before = 0, speedup_after = 0;
                speedup_before = ((baseline_before_time - benchmark_before_time) / baseline_before_time) * 100;
                speedup_after  = ((baseline_after_time  - benchmark_after_time)  / baseline_after_time)  * 100;
                cout << "\r[" 
                    << LowPrecision::get_method_string(LowPrecision::Method::kInt8Ternary) 
                    << "] Speed-Up Increase: " << speedup_before << "% -> " << speedup_after << "% ( " 
                    << (((speedup_after - speedup_before) / abs(speedup_before)) * 100)
                    << "% increase )                ";
            }
            else{
                cout << "\r[" 
                    << LowPrecision::get_method_string(LowPrecision::Method::kInt8Ternary) 
                    << "] Benchmark Time Increase: " << benchmark_before_time << " -> " << benchmark_after_time << " seconds ( " 
                    << (((benchmark_after_time - benchmark_before_time) / benchmark_before_time) * 100)
                    << "% increase )                ";
            }
            cout << endl;
        }
        if (benchmarks.single_mul_api_increasing_size_benchmark_mode & 0x0010){
            benchmark_before_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape, LowPrecision::Method::kInt4ActInt8Weight, disable_progress);
            benchmark_after_time  = run_real_mul_api_benchmark(benchmark_iterations, input_shape_increased, kernel_shape_increased, output_shape_increased, LowPrecision::Method::kInt4ActInt8Weight, disable_progress);
            if (show_speedups){
                double speedup_before = 0, speedup_after = 0;
                speedup_before = ((baseline_before_time - benchmark_before_time) / baseline_before_time) * 100;
                speedup_after  = ((baseline_after_time  - benchmark_after_time)  / baseline_after_time)  * 100;
                cout << "\r[" 
                    << LowPrecision::get_method_string(LowPrecision::Method::kInt4ActInt8Weight) 
                    << "] Speed-Up Increase: " << speedup_before << "% -> " << speedup_after << "% ( " 
                    << (((speedup_after - speedup_before) / abs(speedup_before)) * 100)
                    << "% increase )                ";
            }
            else{
                cout << "\r[" 
                    << LowPrecision::get_method_string(LowPrecision::Method::kInt4ActInt8Weight) 
                    << "] Benchmark Time Increase: " << benchmark_before_time << " -> " << benchmark_after_time << " seconds ( " 
                    << (((benchmark_after_time - benchmark_before_time) / benchmark_before_time) * 100)
                    << "% increase )                ";
            }
            cout << endl;
        }
        if (benchmarks.single_mul_api_increasing_size_benchmark_mode & 0x0020){
            benchmark_before_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape, LowPrecision::Method::kInt4ActInt4Weight, disable_progress);
            benchmark_after_time  = run_real_mul_api_benchmark(benchmark_iterations, input_shape_increased, kernel_shape_increased, output_shape_increased, LowPrecision::Method::kInt4ActInt4Weight, disable_progress);
            if (show_speedups){
                double speedup_before = 0, speedup_after = 0;
                speedup_before = ((baseline_before_time - benchmark_before_time) / baseline_before_time) * 100;
                speedup_after  = ((baseline_after_time  - benchmark_after_time)  / baseline_after_time)  * 100;
                cout << "\r[" 
                    << LowPrecision::get_method_string(LowPrecision::Method::kInt4ActInt4Weight) 
                    << "] Speed-Up Increase: " << speedup_before << "% -> " << speedup_after << "% ( " 
                    << (((speedup_after - speedup_before) / abs(speedup_before)) * 100)
                    << "% increase )                ";
            }
            else{
                cout << "\r[" 
                    << LowPrecision::get_method_string(LowPrecision::Method::kInt4ActInt4Weight) 
                    << "] Benchmark Time Increase: " << benchmark_before_time << " -> " << benchmark_after_time << " seconds ( " 
                    << (((benchmark_after_time - benchmark_before_time) / benchmark_before_time) * 100)
                    << "% increase )                ";
            }
            cout << endl;
        }
        if (benchmarks.single_mul_api_increasing_size_benchmark_mode & 0x0040){
            benchmark_before_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape, LowPrecision::Method::kTernaryActInt8Weight, disable_progress);
            benchmark_after_time  = run_real_mul_api_benchmark(benchmark_iterations, input_shape_increased, kernel_shape_increased, output_shape_increased, LowPrecision::Method::kTernaryActInt8Weight, disable_progress);
            if (show_speedups){
                double speedup_before = 0, speedup_after = 0;
                speedup_before = ((baseline_before_time - benchmark_before_time) / baseline_before_time) * 100;
                speedup_after  = ((baseline_after_time  - benchmark_after_time)  / baseline_after_time)  * 100;
                cout << "\r[" 
                    << LowPrecision::get_method_string(LowPrecision::Method::kTernaryActInt8Weight) 
                    << "] Speed-Up Increase: " << speedup_before << "% -> " << speedup_after << "% ( " 
                    << (((speedup_after - speedup_before) / abs(speedup_before)) * 100)
                    << "% increase )                ";
            }
            else{
                cout << "\r[" 
                    << LowPrecision::get_method_string(LowPrecision::Method::kTernaryActInt8Weight) 
                    << "] Benchmark Time Increase: " << benchmark_before_time << " -> " << benchmark_after_time << " seconds ( " 
                    << (((benchmark_after_time - benchmark_before_time) / benchmark_before_time) * 100)
                    << "% increase )                ";
            }
            cout << endl;
        }
        if (benchmarks.single_mul_api_increasing_size_benchmark_mode & 0x0200){
            benchmark_before_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape, LowPrecision::Method::kTernaryActTernaryWeight, disable_progress);
            benchmark_after_time  = run_real_mul_api_benchmark(benchmark_iterations, input_shape_increased, kernel_shape_increased, output_shape_increased, LowPrecision::Method::kTernaryActTernaryWeight, disable_progress);
            if (show_speedups){
                double speedup_before = 0, speedup_after = 0;
                speedup_before = ((baseline_before_time - benchmark_before_time) / baseline_before_time) * 100;
                speedup_after  = ((baseline_after_time  - benchmark_after_time)  / baseline_after_time)  * 100;
                cout << "\r[" 
                    << LowPrecision::get_method_string(LowPrecision::Method::kTernaryActTernaryWeight) 
                    << "] Speed-Up Increase: " << speedup_before << "% -> " << speedup_after << "% ( " 
                    << (((speedup_after - speedup_before) / abs(speedup_before)) * 100)
                    << "% increase )                ";
            }
            else{
                cout << "\r[" 
                    << LowPrecision::get_method_string(LowPrecision::Method::kTernaryActTernaryWeight) 
                    << "] Benchmark Time Increase: " << benchmark_before_time << " -> " << benchmark_after_time << " seconds ( " 
                    << (((benchmark_after_time - benchmark_before_time) / benchmark_before_time) * 100)
                    << "% increase )                ";
            }
            cout << endl;
        }
        if (benchmarks.single_mul_api_increasing_size_benchmark_mode & 0x0080){
        benchmark_before_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape, LowPrecision::Method::kBinaryActInt8Weight, disable_progress);
        benchmark_after_time  = run_real_mul_api_benchmark(benchmark_iterations, input_shape_increased, kernel_shape_increased, output_shape_increased, LowPrecision::Method::kBinaryActInt8Weight, disable_progress);
            if (show_speedups){
                double speedup_before = 0, speedup_after = 0;
                speedup_before = ((baseline_before_time - benchmark_before_time) / baseline_before_time) * 100;
                speedup_after  = ((baseline_after_time  - benchmark_after_time)  / baseline_after_time)  * 100;
                cout << "\r[" 
                    << LowPrecision::get_method_string(LowPrecision::Method::kBinaryActInt8Weight) 
                    << "] Speed-Up Increase: " << speedup_before << "% -> " << speedup_after << "% ( " 
                    << (((speedup_after - speedup_before) / abs(speedup_before)) * 100)
                    << "% increase )                ";
            }
            else{
                cout << "\r[" 
                    << LowPrecision::get_method_string(LowPrecision::Method::kBinaryActInt8Weight) 
                    << "] Benchmark Time Increase: " << benchmark_before_time << " -> " << benchmark_after_time << " seconds ( " 
                    << (((benchmark_after_time - benchmark_before_time) / benchmark_before_time) * 100)
                    << "% increase )                ";
            }
            cout << endl;
        }
        if (benchmarks.single_mul_api_increasing_size_benchmark_mode & 0x0100){
        benchmark_before_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape, LowPrecision::Method::kBinaryActBinaryWeight, disable_progress);
        benchmark_after_time  = run_real_mul_api_benchmark(benchmark_iterations, input_shape_increased, kernel_shape_increased, output_shape_increased, LowPrecision::Method::kBinaryActBinaryWeight, disable_progress);
            if (show_speedups){
                double speedup_before = 0, speedup_after = 0;
                speedup_before = ((baseline_before_time - benchmark_before_time) / baseline_before_time) * 100;
                speedup_after  = ((baseline_after_time  - benchmark_after_time)  / baseline_after_time)  * 100;
                cout << "\r[" 
                    << LowPrecision::get_method_string(LowPrecision::Method::kBinaryActBinaryWeight) 
                    << "] Speed-Up Increase: " << speedup_before << "% -> " << speedup_after << "% ( " 
                    << (((speedup_after - speedup_before) / abs(speedup_before)) * 100)
                    << "% increase )                ";
            }
            else{
                cout << "\r[" 
                    << LowPrecision::get_method_string(LowPrecision::Method::kBinaryActBinaryWeight) 
                    << "] Benchmark Time Increase: " << benchmark_before_time << " -> " << benchmark_after_time << " seconds ( " 
                    << (((benchmark_after_time - benchmark_before_time) / benchmark_before_time) * 100)
                    << "% increase )                ";
            }
            cout << endl;
        }
        if (benchmarks.single_mul_api_increasing_size_benchmark_mode & 0x0008){
            benchmark_before_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape, kernel_shape, output_shape, LowPrecision::Method::kInt8QuaTernary, disable_progress);
            benchmark_after_time  = run_real_mul_api_benchmark(benchmark_iterations, input_shape_increased, kernel_shape_increased, output_shape_increased, LowPrecision::Method::kInt8QuaTernary, disable_progress);
            if (show_speedups){
                double speedup_before = 0, speedup_after = 0;
                speedup_before = ((baseline_before_time - benchmark_before_time) / baseline_before_time) * 100;
                speedup_after  = ((baseline_after_time  - benchmark_after_time)  / baseline_after_time)  * 100;
                cout << "\r[" 
                    << LowPrecision::get_method_string(LowPrecision::Method::kInt8QuaTernary) 
                    << "] Speed-Up Increase: " << speedup_before << "% -> " << speedup_after << "% ( " 
                    << (((speedup_after - speedup_before) / abs(speedup_before)) * 100)
                    << "% increase )                ";
            }
            else{
                cout << "\r[" 
                    << LowPrecision::get_method_string(LowPrecision::Method::kInt8QuaTernary) 
                    << "] Benchmark Time Increase: " << benchmark_before_time << " -> " << benchmark_after_time << " seconds ( " 
                    << (((benchmark_after_time - benchmark_before_time) / benchmark_before_time) * 100)
                    << "% increase )                ";
            }
            cout << endl;
        }
    }
    if (benchmarks.single_mul_api_different_size_benchmark_enable){
        vector<pair<size_t, size_t>> sizes_v;
        if (LowPrecision::FullyConnected::GetVariableFromEnv( "Sizes" ) == ""){
            cerr << "No 'Sizes' variable is specified. Falling back to default sizes" << endl;
            sizes_v.push_back(pair<size_t, size_t>(512,  512));
            sizes_v.push_back(pair<size_t, size_t>(2048, 512));
            sizes_v.push_back(pair<size_t, size_t>(2048, 2048));
            sizes_v.push_back(pair<size_t, size_t>(4096, 8192));
        }
        else
            sizes_v = extractSizesSingleBatch(LowPrecision::FullyConnected::GetVariableFromEnv( "Sizes" ));

        cout << "Running Single-Batch Mul API Different Size benchmark with below sizes: " << endl;
        for (size_t i = 0; i < sizes_v.size(); i++)
            cout << "\t" << sizes_v[i].first << " x " << sizes_v[i].second << endl;
        
        string time_csv_file_context = "kInt8In8,", speedup_csv_file_context = "kInt8In8,";
        
        if (benchmarks.single_mul_api_different_size_benchmark_mode & 0x0001){
            time_csv_file_context    += "Int8Int4,";
            speedup_csv_file_context += "Int8Int4,";
        }
        if (benchmarks.single_mul_api_different_size_benchmark_mode & 0x0002){
            time_csv_file_context    += "Int8Binary,";
            speedup_csv_file_context += "Int8Binary,";
        }
        if (benchmarks.single_mul_api_different_size_benchmark_mode & 0x0004){
            time_csv_file_context    += "Int8Ternary,";
            speedup_csv_file_context += "Int8Ternary,";
        }
        if (benchmarks.single_mul_api_different_size_benchmark_mode & 0x0010){
            time_csv_file_context    += "Int4ActInt8Weight,";
            speedup_csv_file_context += "Int4ActInt8Weight,";
        }
        if (benchmarks.single_mul_api_different_size_benchmark_mode & 0x0020){
            time_csv_file_context    += "Int4ActInt4Weight,";
            speedup_csv_file_context += "Int4ActInt4Weight,";
        }
        if (benchmarks.single_mul_api_different_size_benchmark_mode & 0x0040){
            time_csv_file_context    += "TernaryActInt8Weight,";
            speedup_csv_file_context += "TernaryActInt8Weight,";
        }
        if (benchmarks.single_mul_api_different_size_benchmark_mode & 0x0200){
            time_csv_file_context    += "TernaryActTernaryWeight,";
            speedup_csv_file_context += "TernaryActTernaryWeight,";
        }
        if (benchmarks.single_mul_api_different_size_benchmark_mode & 0x0080){
            time_csv_file_context    += "BinaryActInt8Weight,";
            speedup_csv_file_context += "BinaryActInt8Weight,";
        }
        if (benchmarks.single_mul_api_different_size_benchmark_mode & 0x0100){
            time_csv_file_context    += "BinaryActBinaryWeight,";
            speedup_csv_file_context += "BinaryActBinaryWeight,";
        }
        if (benchmarks.single_mul_api_different_size_benchmark_mode & 0x0008){
            time_csv_file_context    += "Int8QuaTernary,";
            speedup_csv_file_context += "Int8QuaTernary,";
        }

        time_csv_file_context    += "\n";
        speedup_csv_file_context += "\n";

        for(pair<size_t, size_t> size : sizes_v){
            string size_string;
            size_string += "[";
            size_string += to_string(size.first);
            size_string += "x";
            size_string += to_string(size.second);
            size_string += "]";
            int _num_inputs_current         = size.first,
                _num_outputs_current        = size.second;
            int _input_shape_current[1]     = { _num_inputs_current  },
                _kernel_shape_current[2]    = { _num_outputs_current , _num_inputs_current },
                _output_shape_current[1]    = { _num_outputs_current };
            Shape input_shape_current       = get_shape(_input_shape_current,       1),
                  kernel_shape_current      = get_shape(_kernel_shape_current,      2),
                  output_shape_current      = get_shape(_output_shape_current,      1);
            double baseline_time = 1, benchmark_time = 0;
            
            baseline_time = run_real_ruy_benchmark(benchmark_iterations, input_shape_current, kernel_shape_current, output_shape_current, disable_progress);
            cout << "\r" << size_string << " Baseline Execution Time: " <<  baseline_time << endl;
            time_csv_file_context    += to_string(baseline_time) + ",";
            speedup_csv_file_context += "1,";

            if (benchmarks.single_mul_api_different_size_benchmark_mode & 0x0001){
                benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape_current, kernel_shape_current, output_shape_current, LowPrecision::Method::kInt8Int4, disable_progress);
                double speedup = 0;
                if (benchmark_time >= 0)
                    speedup = ((baseline_time - benchmark_time) / baseline_time) * 100;
                cout << "\r" << size_string
                     << " [" << LowPrecision::get_method_string(LowPrecision::Method::kInt8Int4) << "] " 
                     << "Execution Time: " << benchmark_time << " seconds ( " << speedup << " % speedup )"
                     << "                                         " << endl;
                time_csv_file_context    += to_string(benchmark_time) + ",";
                speedup_csv_file_context += to_string(speedup) + ",";
            }
            if (benchmarks.single_mul_api_different_size_benchmark_mode & 0x0002){
                benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape_current, kernel_shape_current, output_shape_current, LowPrecision::Method::kInt8Binary, disable_progress);
                double speedup = 0;
                if (benchmark_time >= 0)
                    speedup = ((baseline_time - benchmark_time) / baseline_time) * 100;
                cout << "\r" << size_string
                     << " [" << LowPrecision::get_method_string(LowPrecision::Method::kInt8Binary) << "] " 
                     << "Execution Time: " << benchmark_time << " seconds ( " << speedup << " % speedup )"
                     << "                                         " << endl;
                time_csv_file_context    += to_string(benchmark_time) + ",";
                speedup_csv_file_context += to_string(speedup) + ",";
            }
            if (benchmarks.single_mul_api_different_size_benchmark_mode & 0x0004){
                benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape_current, kernel_shape_current, output_shape_current, LowPrecision::Method::kInt8Ternary, disable_progress);
                double speedup = 0;
                if (benchmark_time >= 0)
                    speedup = ((baseline_time - benchmark_time) / baseline_time) * 100;
                cout << "\r" << size_string
                     << " [" << LowPrecision::get_method_string(LowPrecision::Method::kInt8Ternary) << "] " 
                     << "Execution Time: " << benchmark_time << " seconds ( " << speedup << " % speedup )"
                     << "                                         " << endl;
                time_csv_file_context    += to_string(benchmark_time) + ",";
                speedup_csv_file_context += to_string(speedup) + ",";
            }
            if (benchmarks.single_mul_api_different_size_benchmark_mode & 0x0010){
                benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape_current, kernel_shape_current, output_shape_current, LowPrecision::Method::kInt4ActInt8Weight, disable_progress);
                double speedup = 0;
                if (benchmark_time >= 0)
                    speedup = ((baseline_time - benchmark_time) / baseline_time) * 100;
                cout << "\r" << size_string
                     << " [" << LowPrecision::get_method_string(LowPrecision::Method::kInt4ActInt8Weight) << "] " 
                     << "Execution Time: " << benchmark_time << " seconds ( " << speedup << " % speedup )"
                     << "                                         " << endl;
                time_csv_file_context    += to_string(benchmark_time) + ",";
                speedup_csv_file_context += to_string(speedup) + ",";
            }
            if (benchmarks.single_mul_api_different_size_benchmark_mode & 0x0020){
                benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape_current, kernel_shape_current, output_shape_current, LowPrecision::Method::kInt4ActInt4Weight, disable_progress);
                double speedup = 0;
                if (benchmark_time >= 0)
                    speedup = ((baseline_time - benchmark_time) / baseline_time) * 100;
                cout << "\r" << size_string
                     << " [" << LowPrecision::get_method_string(LowPrecision::Method::kInt4ActInt4Weight) << "] " 
                     << "Execution Time: " << benchmark_time << " seconds ( " << speedup << " % speedup )"
                     << "                                         " << endl;
                time_csv_file_context    += to_string(benchmark_time) + ",";
                speedup_csv_file_context += to_string(speedup) + ",";
            }
            if (benchmarks.single_mul_api_different_size_benchmark_mode & 0x0040){
                benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape_current, kernel_shape_current, output_shape_current, LowPrecision::Method::kTernaryActInt8Weight, disable_progress);
                double speedup = 0;
                if (benchmark_time >= 0)
                    speedup = ((baseline_time - benchmark_time) / baseline_time) * 100;
                cout << "\r" << size_string
                     << " [" << LowPrecision::get_method_string(LowPrecision::Method::kTernaryActInt8Weight) << "] " 
                     << "Execution Time: " << benchmark_time << " seconds ( " << speedup << " % speedup )"
                     << "                                         " << endl;
                time_csv_file_context    += to_string(benchmark_time) + ",";
                speedup_csv_file_context += to_string(speedup) + ",";
            }
            if (benchmarks.single_mul_api_different_size_benchmark_mode & 0x0200){
                benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape_current, kernel_shape_current, output_shape_current, LowPrecision::Method::kTernaryActTernaryWeight, disable_progress);
                double speedup = 0;
                if (benchmark_time >= 0)
                    speedup = ((baseline_time - benchmark_time) / baseline_time) * 100;
                cout << "\r" << size_string
                     << " [" << LowPrecision::get_method_string(LowPrecision::Method::kTernaryActTernaryWeight) << "] " 
                     << "Execution Time: " << benchmark_time << " seconds ( " << speedup << " % speedup )"
                     << "                                         " << endl;
                time_csv_file_context    += to_string(benchmark_time) + ",";
                speedup_csv_file_context += to_string(speedup) + ",";
            }
            if (benchmarks.single_mul_api_different_size_benchmark_mode & 0x0080){
                benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape_current, kernel_shape_current, output_shape_current, LowPrecision::Method::kBinaryActInt8Weight, disable_progress);
                double speedup = 0;
                if (benchmark_time >= 0)
                    speedup = ((baseline_time - benchmark_time) / baseline_time) * 100;
                cout << "\r" << size_string
                     << " [" << LowPrecision::get_method_string(LowPrecision::Method::kBinaryActInt8Weight) << "] " 
                     << "Execution Time: " << benchmark_time << " seconds ( " << speedup << " % speedup )"
                     << "                                         " << endl;
                time_csv_file_context    += to_string(benchmark_time) + ",";
                speedup_csv_file_context += to_string(speedup) + ",";
            }
            if (benchmarks.single_mul_api_different_size_benchmark_mode & 0x0100){
                benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape_current, kernel_shape_current, output_shape_current, LowPrecision::Method::kBinaryActBinaryWeight, disable_progress);
                double speedup = 0;
                if (benchmark_time >= 0)
                    speedup = ((baseline_time - benchmark_time) / baseline_time) * 100;
                cout << "\r" << size_string
                     << " [" << LowPrecision::get_method_string(LowPrecision::Method::kBinaryActBinaryWeight) << "] " 
                     << "Execution Time: " << benchmark_time << " seconds ( " << speedup << " % speedup )"
                     << "                                         " << endl;
                time_csv_file_context    += to_string(benchmark_time) + ",";
                speedup_csv_file_context += to_string(speedup) + ",";
            }
            if (benchmarks.single_mul_api_different_size_benchmark_mode & 0x0008){
                benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape_current, kernel_shape_current, output_shape_current, LowPrecision::Method::kInt8QuaTernary, disable_progress);
                double speedup = 0;
                if (benchmark_time >= 0)
                    speedup = ((baseline_time - benchmark_time) / baseline_time) * 100;
                cout << "\r" << size_string
                     << " [" << LowPrecision::get_method_string(LowPrecision::Method::kInt8QuaTernary) << "] " 
                     << "Execution Time: " << benchmark_time << " seconds ( " << speedup << " % speedup )"
                     << "                                         " << endl;
                time_csv_file_context    += to_string(benchmark_time) + ",";
                speedup_csv_file_context += to_string(speedup) + ",";
            }

            time_csv_file_context    += "\n";
            speedup_csv_file_context += "\n";
        }
        ofstream time_csv_file(benchmarks.single_mul_api_different_size_benchmark_time_path, ios::out | ios::trunc);
        ofstream speedup_csv_file(benchmarks.single_mul_api_different_size_benchmark_speedup_path, ios::out | ios::trunc);

        if (!time_csv_file)
            cerr << "Could not open file '"<< benchmarks.single_mul_api_different_size_benchmark_time_path <<"' for writing." << endl;
        else{
            time_csv_file << time_csv_file_context;
            time_csv_file.close();
        }
        if (!speedup_csv_file)
            cerr << "Could not open file '"<< benchmarks.single_mul_api_different_size_benchmark_speedup_path <<"' for writing." << endl;
        else{
            speedup_csv_file << speedup_csv_file_context;
            speedup_csv_file.close();
        }
    }
    if (benchmarks.multi_mul_api_different_size_benchmark_enable){
        vector<tuple<size_t, size_t, size_t>> sizes_v;
        if (LowPrecision::FullyConnected::GetVariableFromEnv( "Sizes" ) == ""){
            cerr << "No 'Sizes' variable is specified. Falling back to default sizes" << endl;
            sizes_v.push_back(tuple<size_t, size_t, size_t>(16, 512,  512));
            sizes_v.push_back(tuple<size_t, size_t, size_t>(16, 2048, 512));
            sizes_v.push_back(tuple<size_t, size_t, size_t>(16, 2048, 2048));
            sizes_v.push_back(tuple<size_t, size_t, size_t>(16, 4096, 8192));
        }
        else
            sizes_v = extractSizesMultiBatch(LowPrecision::FullyConnected::GetVariableFromEnv( "Sizes" ));

        cout << "Running Multi-Batch Mul API Different Size benchmark with below sizes: " << endl;
        for (size_t i = 0; i < sizes_v.size(); i++)
            cout << "\t" << get<0>(sizes_v[i]) << " x " << get<1>(sizes_v[i]) << " x " << get<2>(sizes_v[i]) << endl;
        
        string time_csv_file_context = "kInt8In8,", speedup_csv_file_context = "kInt8In8,";
        
        if (benchmarks.multi_mul_api_different_size_benchmark_mode & 0x0001){
            time_csv_file_context    += "Int8Int4,";
            speedup_csv_file_context += "Int8Int4,";
        }
        if (benchmarks.multi_mul_api_different_size_benchmark_mode & 0x0002){
            time_csv_file_context    += "Int8Binary,";
            speedup_csv_file_context += "Int8Binary,";
        }
        if (benchmarks.multi_mul_api_different_size_benchmark_mode & 0x0004){
            time_csv_file_context    += "Int8Ternary,";
            speedup_csv_file_context += "Int8Ternary,";
        }
        if (benchmarks.multi_mul_api_different_size_benchmark_mode & 0x0010){
            time_csv_file_context    += "Int4ActInt8Weight,";
            speedup_csv_file_context += "Int4ActInt8Weight,";
        }
        if (benchmarks.multi_mul_api_different_size_benchmark_mode & 0x0020){
            time_csv_file_context    += "Int4ActInt4Weight,";
            speedup_csv_file_context += "Int4ActInt4Weight,";
        }
        if (benchmarks.multi_mul_api_different_size_benchmark_mode & 0x0040){
            time_csv_file_context    += "TernaryActInt8Weight,";
            speedup_csv_file_context += "TernaryActInt8Weight,";
        }
        if (benchmarks.multi_mul_api_different_size_benchmark_mode & 0x0200){
            time_csv_file_context    += "TernaryActTernaryWeight,";
            speedup_csv_file_context += "TernaryActTernaryWeight,";
        }

        time_csv_file_context    += "\n";
        speedup_csv_file_context += "\n";

        for(tuple<size_t, size_t, size_t> size : sizes_v){
            string size_string;
            size_string += "[";
            size_string += to_string(get<0>(size));
            size_string += "x";
            size_string += to_string(get<1>(size));
            size_string += "x";
            size_string += to_string(get<2>(size));
            size_string += "]";
            int _num_batches_current        = get<0>(size),
                _num_inputs_current         = get<1>(size),
                _num_outputs_current        = get<2>(size);
            int _input_shape_current[2]     = { _num_batches_current , _num_inputs_current  },
                _kernel_shape_current[2]    = { _num_outputs_current , _num_inputs_current  },
                _output_shape_current[2]    = { _num_batches_current , _num_outputs_current };
            Shape input_shape_current       = get_shape(_input_shape_current,       2),
                  kernel_shape_current      = get_shape(_kernel_shape_current,      2),
                  output_shape_current      = get_shape(_output_shape_current,      2);
            double baseline_time = 1, benchmark_time = 0;
            
            baseline_time = run_real_ruy_benchmark(benchmark_iterations, input_shape_current, kernel_shape_current, output_shape_current, disable_progress);
            cout << "\r" << size_string << " Baseline Execution Time: " <<  baseline_time << endl;
            time_csv_file_context    += to_string(baseline_time) + ",";
            speedup_csv_file_context += "1,";

            if (benchmarks.multi_mul_api_different_size_benchmark_mode & 0x0001){
                benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape_current, kernel_shape_current, output_shape_current, LowPrecision::Method::kInt8Int4, disable_progress);
                double speedup = 0;
                if (benchmark_time >= 0)
                    speedup = ((baseline_time - benchmark_time) / baseline_time) * 100;
                cout << "\r" << size_string
                     << " [" << LowPrecision::get_method_string(LowPrecision::Method::kInt8Int4) << "] " 
                     << "Execution Time: " << benchmark_time << " seconds ( " << speedup << " % speedup )"
                     << "                                         " << endl;
                time_csv_file_context    += to_string(benchmark_time) + ",";
                speedup_csv_file_context += to_string(speedup) + ",";
            }
            if (benchmarks.multi_mul_api_different_size_benchmark_mode & 0x0002){
                benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape_current, kernel_shape_current, output_shape_current, LowPrecision::Method::kInt8Binary, disable_progress);
                double speedup = 0;
                if (benchmark_time >= 0)
                    speedup = ((baseline_time - benchmark_time) / baseline_time) * 100;
                cout << "\r" << size_string
                     << " [" << LowPrecision::get_method_string(LowPrecision::Method::kInt8Binary) << "] " 
                     << "Execution Time: " << benchmark_time << " seconds ( " << speedup << " % speedup )"
                     << "                                         " << endl;
                time_csv_file_context    += to_string(benchmark_time) + ",";
                speedup_csv_file_context += to_string(speedup) + ",";
            }
            if (benchmarks.multi_mul_api_different_size_benchmark_mode & 0x0004){
                benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape_current, kernel_shape_current, output_shape_current, LowPrecision::Method::kInt8Ternary, disable_progress);
                double speedup = 0;
                if (benchmark_time >= 0)
                    speedup = ((baseline_time - benchmark_time) / baseline_time) * 100;
                cout << "\r" << size_string
                     << " [" << LowPrecision::get_method_string(LowPrecision::Method::kInt8Ternary) << "] " 
                     << "Execution Time: " << benchmark_time << " seconds ( " << speedup << " % speedup )"
                     << "                                         " << endl;
                time_csv_file_context    += to_string(benchmark_time) + ",";
                speedup_csv_file_context += to_string(speedup) + ",";
            }
            if (benchmarks.multi_mul_api_different_size_benchmark_mode & 0x0010){
                benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape_current, kernel_shape_current, output_shape_current, LowPrecision::Method::kInt4ActInt8Weight, disable_progress);
                double speedup = 0;
                if (benchmark_time >= 0)
                    speedup = ((baseline_time - benchmark_time) / baseline_time) * 100;
                cout << "\r" << size_string
                     << " [" << LowPrecision::get_method_string(LowPrecision::Method::kInt4ActInt8Weight) << "] " 
                     << "Execution Time: " << benchmark_time << " seconds ( " << speedup << " % speedup )"
                     << "                                         " << endl;
                time_csv_file_context    += to_string(benchmark_time) + ",";
                speedup_csv_file_context += to_string(speedup) + ",";
            }
            if (benchmarks.multi_mul_api_different_size_benchmark_mode & 0x0020){
                benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape_current, kernel_shape_current, output_shape_current, LowPrecision::Method::kInt4ActInt4Weight, disable_progress);
                double speedup = 0;
                if (benchmark_time >= 0)
                    speedup = ((baseline_time - benchmark_time) / baseline_time) * 100;
                cout << "\r" << size_string
                     << " [" << LowPrecision::get_method_string(LowPrecision::Method::kInt4ActInt4Weight) << "] " 
                     << "Execution Time: " << benchmark_time << " seconds ( " << speedup << " % speedup )"
                     << "                                         " << endl;
                time_csv_file_context    += to_string(benchmark_time) + ",";
                speedup_csv_file_context += to_string(speedup) + ",";
            }
            if (benchmarks.multi_mul_api_different_size_benchmark_mode & 0x0040){
                benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape_current, kernel_shape_current, output_shape_current, LowPrecision::Method::kTernaryActInt8Weight, disable_progress);
                double speedup = 0;
                if (benchmark_time >= 0)
                    speedup = ((baseline_time - benchmark_time) / baseline_time) * 100;
                cout << "\r" << size_string
                     << " [" << LowPrecision::get_method_string(LowPrecision::Method::kTernaryActInt8Weight) << "] " 
                     << "Execution Time: " << benchmark_time << " seconds ( " << speedup << " % speedup )"
                     << "                                         " << endl;
                time_csv_file_context    += to_string(benchmark_time) + ",";
                speedup_csv_file_context += to_string(speedup) + ",";
            }
            if (benchmarks.multi_mul_api_different_size_benchmark_mode & 0x0200){
                benchmark_time = run_real_mul_api_benchmark(benchmark_iterations, input_shape_current, kernel_shape_current, output_shape_current, LowPrecision::Method::kTernaryActTernaryWeight, disable_progress);
                double speedup = 0;
                if (benchmark_time >= 0)
                    speedup = ((baseline_time - benchmark_time) / baseline_time) * 100;
                cout << "\r" << size_string
                     << " [" << LowPrecision::get_method_string(LowPrecision::Method::kTernaryActTernaryWeight) << "] " 
                     << "Execution Time: " << benchmark_time << " seconds ( " << speedup << " % speedup )"
                     << "                                         " << endl;
                time_csv_file_context    += to_string(benchmark_time) + ",";
                speedup_csv_file_context += to_string(speedup) + ",";
            }

            time_csv_file_context    += "\n";
            speedup_csv_file_context += "\n";
        }
        ofstream time_csv_file(benchmarks.multi_mul_api_different_size_benchmark_time_path, ios::out | ios::trunc);
        ofstream speedup_csv_file(benchmarks.multi_mul_api_different_size_benchmark_speedup_path, ios::out | ios::trunc);

        if (!time_csv_file)
            cerr << "Could not open file '"<< benchmarks.multi_mul_api_different_size_benchmark_time_path <<"' for writing." << endl;
        else{
            time_csv_file << time_csv_file_context;
            time_csv_file.close();
        }
        if (!speedup_csv_file)
            cerr << "Could not open file '"<< benchmarks.multi_mul_api_different_size_benchmark_speedup_path <<"' for writing." << endl;
        else{
            speedup_csv_file << speedup_csv_file_context;
            speedup_csv_file.close();
        }
    }
    return;
}

vector<pair<size_t, size_t>> extractSizesSingleBatch(string str){
    vector<pair<size_t, size_t>> result;
    pair<size_t, size_t> current_size;
    bool current_pair_n = false;
    string current = ""; 
    for(int i = 0; i < str.size(); i++){
        if(str[i] == 'x'){
            if(current != string()){
                current_size.first = stoi(current);
                current = "";
            }
            continue;
        }
        if(str[i] == ','){
            if(current != string()){
                current_size.second = stoi(current);
                result.push_back(current_size);
                current = "";
                current_size.first  = 0;
                current_size.second = 0;
            }
            continue;
        }
        current += str[i];
    }
    if(current.size() != 0){
        current_size.second = stoi(current);
        result.push_back(current_size);
    }
    return result;
}

vector<tuple<size_t, size_t, size_t>> extractSizesMultiBatch(string str){
    vector<tuple<size_t, size_t, size_t>> result;
    tuple<size_t, size_t, size_t> current_size;
    int current_pair_n = 0;
    string current = ""; 
    for(int i = 0; i < str.size(); i++){
        if(str[i] == 'x'){
            if(current != string()){
                if (current_pair_n == 0)
                    get<0>(current_size) = stoi(current);
                if (current_pair_n == 1)
                    get<1>(current_size) = stoi(current);
                current_pair_n++;
                current = "";
            }
            continue;
        }
        if(str[i] == ','){
            if(current != string()){
                get<2>(current_size) = stoi(current);
                result.push_back(current_size);
                current = "";
                current_pair_n = 0;
            }
            continue;
        }
        current += str[i];
    }
    if(current.size() != 0){
        get<2>(current_size) = stoi(current);
        result.push_back(current_size);
    }
    return result;
}




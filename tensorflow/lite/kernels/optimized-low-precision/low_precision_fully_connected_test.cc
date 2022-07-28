#include "low_precision_fully_connected_benchmark.h"
#include "ruy/ruy.h"
#include "ruy/context.h"
#include <stdio.h>
#include <string>

using namespace std;
using namespace LowPrecision;
using namespace LowPrecision::FullyConnected;

#define DEBUG false
#define PRINT_ACTIVATIONS false
#define PRINT_MUL_OUTPUT false
#define PRINT_MUL_MB_OUTPUT false
#define PRINT_VALUES_IN_HEX false
#define PRINT_VALUES false
#define PRINT_KERNEL_IN_HEX false
#define PRINT_KERNEL false
#define PRINT_FILTER_IN_HEX false
#define PRINT_FILTER false

void run_mul_api_tests(LowPrecision::Method method){
    int num_spaces = 40 - string(LowPrecision::get_method_string(method)).length();
    vector<char> spaces_vec(num_spaces, ' ');
    string spaces(spaces_vec.begin(), spaces_vec.end());
    // Setting Constant Values
    int num_batch               = 64,
        num_inputs              = 5,
        num_output              = 4;

    // Creating Size Arrays
    int _input_shape[2]         = {     1     , num_inputs },
        _activation_shape[2]    = {     1     , num_inputs },
        _input_shape_MB[2]      = { num_batch , num_inputs },
        _activation_MB_shape[2] = { num_batch , num_inputs},
        _kernel_shape[2]        = { num_output, num_inputs },
        _filter_shape[2]        = { num_output, num_inputs},
        _output_shape[2]        = {     1     , num_output },
        _output_shape_MB[2]     = { num_batch , num_output };
    
    // Transforming Input Shapes Based on Method
    LowPrecision::FullyConnected::TransformInputShape (method, _activation_shape,    2);
    LowPrecision::FullyConnected::TransformInputShape (method, _activation_MB_shape, 2);

    // Transforming Filter Shapes Based on Method
    LowPrecision::FullyConnected::TransformFilterShape(method, _filter_shape,        2);

    // Creating Shapes
    Shape input_shape           = get_shape(_input_shape,           2),
          activation_shape      = get_shape(_activation_shape,      2),
          input_shape_MB        = get_shape(_input_shape_MB,        2),
          activation_shape_MB   = get_shape(_activation_MB_shape,   2),
          kernel_shape          = get_shape(_kernel_shape,          2),
          filter_shape          = get_shape(_filter_shape,          2),
          output_shape          = get_shape(_output_shape,          2),
          output_shape_MB       = get_shape(_output_shape_MB,       2);

    // cout << "Transformed Input Shapes " 
    //      << LowPrecision::get_shape_string(activation_shape) << " "
    //      << LowPrecision::get_shape_string(activation_shape_MB)
    //      << endl;
    // cout << "Transformed Filter Shapes " 
    //      << LowPrecision::get_shape_string(filter_shape)
    //      << endl;

    // Allocating Matrices
    int8_t*  input_data         = LowPrecision::allocate<int8_t>(input_shape.flatsize);
    int8_t*  activation_data    = LowPrecision::allocate<int8_t>(activation_shape.flatsize);
    int8_t*  input_data_MB      = LowPrecision::allocate<int8_t>(input_shape_MB.flatsize);
    int8_t*  activation_data_MB = LowPrecision::allocate<int8_t>(activation_shape_MB.flatsize);
    int8_t*  kernel_data        = LowPrecision::allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  filter_data        = LowPrecision::allocate<int8_t>(filter_shape.flatsize);
    int32_t* output_data        = LowPrecision::allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_MB     = LowPrecision::allocate<int32_t>(output_shape_MB.flatsize);

    // Processing Kernel to Filter
    LowPrecision::Status filter_status = LowPrecision::FullyConnected::QuantizeFilter(method, kernel_data, kernel_shape, filter_data, LowPrecision::MemLayout::kRowMajor);

    if (LowPrecision::mask_out_source(filter_status) == LowPrecision::Status::Success)
        cout << LowPrecision::get_method_string(method) << " Mul API Filter Processing Test" << spaces.substr(6) << "=> \033[1m\033[32mPASSED\033[0m" << endl;
    else
        cout << LowPrecision::get_method_string(method) << " Mul API Filter Processing Test" << spaces.substr(6) << "=> \033[1m\033[31mFAILED\033[0m (Sourcce: "
                                                        << LowPrecision::get_status_string(LowPrecision::mask_out_status(filter_status))
                                                        << " | Status: "
                                                        << LowPrecision::get_status_string(LowPrecision::mask_out_source(filter_status))
                                                        << ")" << endl;

    // Creating Filter Matrix
    LowPrecision::Matrix filter_matrix;
    filter_matrix.setDataAndScratchpadAndShape(nullptr, filter_data, kernel_shape);
    filter_matrix.setNeedScratchpad();
    filter_matrix.setScratchpadValid();
    filter_matrix.setMemLayout(LowPrecision::MemLayout::kRowMajor);

    ///////////////////////////////////////////////////////////////
    //////////////////   Single Batch API Test   //////////////////
    ///////////////////////////////////////////////////////////////

    // Creating Single Batch Input Matrix
    LowPrecision::Matrix input_matrix;
    input_matrix.setDataAndScratchpadAndShape(input_data, activation_data, input_shape);
    input_matrix.setNeedScratchpad();
    input_matrix.setMemLayout(LowPrecision::MemLayout::kRowMajor);

    // Creating Single Batch Output Matrix
    LowPrecision::Matrix output_matrix;
    output_matrix.setDataAndScratchpadAndShape(output_data, nullptr, output_shape);
    output_matrix.setMemLayout(LowPrecision::MemLayout::kRowMajor);
    
    // Single Batch Multiplication
    LowPrecision::Status single_ret = LowPrecision::FullyConnected::Mul(input_matrix, filter_matrix, output_matrix, method);

    if (LowPrecision::mask_out_source(single_ret) == LowPrecision::Status::Success)
        cout << LowPrecision::get_method_string(method) << " Mul API Single-Batch Test" << spaces.substr(1) << "=> \033[1m\033[32mPASSED\033[0m" << endl;
    else
        cout << LowPrecision::get_method_string(method) << " Mul API Single-Batch Test" << spaces.substr(1) << "=> \033[1m\033[31mFAILED\033[0m (Sourcce: "
                                                        << LowPrecision::get_status_string(LowPrecision::mask_out_status(single_ret))
                                                        << " | Status: "
                                                        << LowPrecision::get_status_string(LowPrecision::mask_out_source(single_ret))
                                                        << ")" << endl;

    ///////////////////////////////////////////////////////////////
    /////////////////    Multi Batch API Test    //////////////////
    ///////////////////////////////////////////////////////////////

    // Creating Multi Batch Input Matrix
    LowPrecision::Matrix input_MB_matrix;
    input_MB_matrix.setDataAndScratchpadAndShape(input_data_MB, activation_data_MB, input_shape_MB);
    input_MB_matrix.setNeedScratchpad();
    input_MB_matrix.setMemLayout(LowPrecision::MemLayout::kRowMajor);

    // Creating Multi Batch Output Matrix
    LowPrecision::Matrix output_MB_matrix;
    output_MB_matrix.setDataAndScratchpadAndShape(output_data_MB, nullptr, output_shape_MB);
    output_MB_matrix.setMemLayout(LowPrecision::MemLayout::kRowMajor);

    // Multi Batch Multiplication
    LowPrecision::Status multi_ret = LowPrecision::FullyConnected::Mul(input_MB_matrix, filter_matrix, output_MB_matrix, method);

    if (LowPrecision::mask_out_source(multi_ret) == LowPrecision::Status::Success)
        cout << LowPrecision::get_method_string(method) << " Mul API Multi-Batch Test" << spaces << "=> \033[1m\033[32mPASSED\033[0m" << endl;
    else
        cout << LowPrecision::get_method_string(method) << " Mul API Multi-Batch Test" << spaces << "=> \033[1m\033[31mFAILED\033[0m (Sourcce: "
                                                        << LowPrecision::get_status_string(LowPrecision::mask_out_status(multi_ret))
                                                        << " | Status: "
                                                        << LowPrecision::get_status_string(LowPrecision::mask_out_source(multi_ret))
                                                        << ")" << endl;

    // Deallication of created pointers
    LowPrecision::deallocate(input_data);
    LowPrecision::deallocate(activation_data);
    LowPrecision::deallocate(input_data_MB);
    LowPrecision::deallocate(activation_data_MB);
    LowPrecision::deallocate(kernel_data);
    LowPrecision::deallocate(filter_data);
    LowPrecision::deallocate(output_data);
    LowPrecision::deallocate(output_data_MB);
}

void run_i8i4_tests(
    const int* _template,
    const int8_t* _answers,
    const int kernel_fill_mode,
    const int num_inputs,
    const int num_output,
    const int num_batch
    ){
    int num_padding = (num_inputs % 32)?(32 - (num_inputs % 32)):(0);
    int _input_shape[1]       = { num_inputs },
        _input_pad_shape[1]   = { num_inputs + num_padding },
        _input_shape_MB[2]    = { num_batch , num_inputs },
        _kernel_shape[2]      = { num_output, num_inputs },
        _kernel_pad_shape[2]  = { num_output, num_inputs + num_padding },
        _filter_shape[2]      = { num_output, num_inputs },
        _output_shape[1]      = { num_output },
        _output_shape_MB[2]   = { num_batch , num_output };

    Shape input_shape         = get_shape(_input_shape,        1),
          input_pad_shape     = get_shape(_input_pad_shape,    1),
          input_shape_MB      = get_shape(_input_shape_MB,     2),
          kernel_shape        = get_shape(_kernel_shape,       2),
          kernel_pad_shape    = get_shape(_kernel_pad_shape,   2),
          filter_shape        = get_shape(_filter_shape,       2),
          output_shape        = get_shape(_output_shape,       1),
          output_shape_MB     = get_shape(_output_shape_MB,    2);
    
    filter_shape.flatsize = LowPrecision::FullyConnected::Int4::TransformFilterShape(filter_shape.size, filter_shape.number_dims);
    
    int8_t*  input_data       = allocate<int8_t>(input_shape.flatsize);
    int8_t*  input_pad_data   = allocate<int8_t>(input_pad_shape.flatsize);
    int8_t*  input_data_MB    = allocate<int8_t>(input_shape_MB.flatsize);
    int8_t*  input_pack_MB    = allocate<int8_t>(input_shape_MB.flatsize);
    int8_t*  kernel_data      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  kernel_data_C    = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  filter_data      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output_data      = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_MB   = allocate<int32_t>(output_shape_MB.flatsize);
    int32_t* output_data_R    = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_R_MB = allocate<int32_t>(output_shape_MB.flatsize);

    for (int i = 0; i < num_inputs; i++)
        input_data[i] = 1;
    
    for (int i = 0; i < num_batch; i++)
        for (int j = 0; j < num_inputs; j++)
            input_data_MB[i * num_inputs + j] = 1;
    
    if(kernel_fill_mode == 0){
        for (int i = 0; i < kernel_shape.size[0]; i++)
            for (int j = 0; j < kernel_shape.size[1]; j++)
                kernel_data[i * kernel_shape.size[1] + j]   = _template[j % 32];

        for (int i = 0; i < kernel_shape.size[1]; i++)
            for (int j = 0; j < kernel_shape.size[0]; j++)
                kernel_data_C[i * kernel_shape.size[0] + j] = _template[i % 32];
    }
    else if (kernel_fill_mode == 1){
        zero_vector(kernel_data, kernel_shape.flatsize);
        zero_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 2){
        one_vector(kernel_data, kernel_shape.flatsize);
        one_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 3){
        minus_one_vector(kernel_data, kernel_shape.flatsize);
        minus_one_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 4){
        two_vector(kernel_data, kernel_shape.flatsize);
        two_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 5){
        minus_two_vector(kernel_data, kernel_shape.flatsize);
        minus_two_vector(kernel_data_C, kernel_shape.flatsize);
    }

#if PRINT_VALUES
    cout << "Kernel = [" << endl;
    for (int i = 0; i < kernel_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < kernel_shape.size[1]; j++)
            cout << ((int)kernel_data[i * kernel_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "Kernel(0x) = [" << hex << endl;
    for (int i = 0; i < kernel_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < kernel_shape.size[1]; j++)
            cout << (((int)kernel_data[i * kernel_shape.size[1] + j]) & 0x0f) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
#if PRINT_VALUES
    cout << "Kernel_C = [" << endl;
    for (int j = 0; j < kernel_shape.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < kernel_shape.size[1]; i++)
            cout << ((int)kernel_data_C[i * kernel_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "Kernel_C(0x) = [" << hex << endl;
    for (int j = 0; j < kernel_shape.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < kernel_shape.size[1]; i++)
            cout << (((int)kernel_data_C[i * kernel_shape.size[0] + j]) & 0x0f) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif

//     kernel_pad_data = LowPrecision::FullyConnected::PaddingWeightsIfNeeded(kernel_data, kernel_shape, LowPrecision::Method::kInt8Int4);
//     cout << "LowPrecision::FullyConnected::PaddingWeightsIfNeeded Return\t\t\t=> " << ((kernel_pad_data == nullptr)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((kernel_pad_data == nullptr)?("FAILED"):("PASSED")) << "\033[0m" << endl;
// #if PRINT_VALUES
//     cout << "Kernel_Padded = [" << endl;
//     for (int i = 0; i < kernel_pad_shape.size[0]; i++){
//         cout << "\t[";
//         for (int j = 0; j < kernel_pad_shape.size[1]; j++)
//             cout << ((int)kernel_pad_data[i * kernel_pad_shape.size[1] + j]) << ", ";
//         cout << "]," << endl;
//     }
//     cout << "]";
//     cout << endl;
// #endif

    Status ret = LowPrecision::FullyConnected::Int4::QuantizeInput(input_data, input_shape, input_pad_data, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::Int4::QuantizeInput-RowMajor Return Status \t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    // if (filter_shape.flatsize <= 36 * 36){
    //     bool QFTI4RMPassed = true;
    //     for (int i = 0 ; i < filter_shape.flatsize ; i++){
    //         // if (_answers[i] != filter_data[i])
    //             // cout << i << ": " << (int)_answers[i] << "->" << (int)filter_data[i] << endl; 
    //         QFTI4RMPassed &= (_answers[i] == filter_data[i]);
    //     }
    //     cout << "LowPrecision::FullyConnected::Int4::QuantizeInput-RowMajor    \t\t\t=> " << ((QFTI4RMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    // }

#if PRINT_VALUES || PRINT_ACTIVATIONS
    cout << LowPrecision::get_shape_string(input_pad_shape) << " [" << endl << "\t";
    for (int i = 0; i < input_pad_shape.size[0]; i++){
        cout << ((int)input_pad_data[i]) << ", ";
    }
    cout << endl;
    cout << "]";
    cout << endl;
#endif

    ret = LowPrecision::FullyConnected::Int4::QuantizeFilter(kernel_data, kernel_shape, filter_data, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::Int4::QuantizeFilter-RowMajor Return Status \t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 36 * 36){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++){
            if (_answers[i] != filter_data[i])
                cout << i << ": " << (int)_answers[i] << "->" << (int)filter_data[i] << endl; 
            QFTI4RMPassed &= (_answers[i] == filter_data[i]);
        }
        cout << "LowPrecision::FullyConnected::Int4::QuantizeFilter-RowMajor    \t\t\t=> " << ((QFTI4RMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_KERNEL
    cout << "[" << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((int)filter_data[i * filter_shape.size[1] + j]) << "(" << ((int)_answers[i * filter_shape.size[1] + j]) << ")" << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_KERNEL_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((((uint8_t)filter_data[i * filter_shape.size[1] + j]) < 16)?("0"):("")) << (((int)filter_data[i * filter_shape.size[1] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    ret = LowPrecision::FullyConnected::Int4::QuantizeFilter(kernel_data, kernel_shape, filter_data, MemLayout::kColumnMajor);
    
    cout << "LowPrecision::FullyConnected::Int4::QuantizeFilter-ColumnMajor Return Status \t=> " << ((ret)?((ret == LowPrecision::Status::WrongMemLayout)?("\033[1m\033[33m"):("\033[1m\033[31m")):("\033[1m\033[32m")) << ((ret)?((ret == LowPrecision::Status::WrongMemLayout)?("NO_SUPPORT"):("FAILED")):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 36 * 36 && !ret){
        bool QFTI4CMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4CMPassed &= (_answers[i] == filter_data[i]);
        cout << "LowPrecision::FullyConnected::Int4::QuantizeFilter-ColumnMajor \t\t\t=> " << ((QFTI4CMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4CMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_KERNEL
    cout << "[" << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_KERNEL_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((((uint8_t)filter_data[i * filter_shape.size[1] + j]) < 16)?("0"):("")) << (((int)filter_data[i * filter_shape.size[1] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif

    doLowPrecisionPack(input_data_MB, input_pack_MB, input_shape_MB.size[0], input_shape_MB.size[1]);

#if PRINT_VALUES || PRINT_ACTIVATIONS
    cout << "[" << endl;
    for (int i = 0; i < input_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape_MB.size[1]; j++)
            cout << ((int)input_data_MB[i * input_shape_MB.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES || PRINT_ACTIVATIONS
    cout << "[" << endl;
    for (int i = 0; i < input_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape_MB.size[1]; j++)
            cout << ((int)input_pack_MB[i * input_shape_MB.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < input_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape_MB.size[1]; j++)
            cout << (uint8_t)input_pack_MB[i * input_shape_MB.size[1] + j] << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    minus_one_vector(output_data_R, output_shape.flatsize);

    ruy::Matrix<int8_t> ruy_lhs;
    ruy::Matrix<int8_t> ruy_rhs;
    ruy::Matrix<int8_t> ruy_rhs_C;
    ruy::Matrix<int32_t> ruy_dst;

    // Create lhs
    ruy::MakeSimpleLayout(1, input_shape.size[0], ruy::Order::kColMajor,
                              ruy_lhs.mutable_layout());
    ruy_lhs.set_data(input_data);
    // Create rhs
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kColMajor,
                              ruy_rhs.mutable_layout());
    ruy_rhs.set_data(kernel_data);
    ruy_rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create rhs
    // ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kRowMajor,
    //                           ruy_rhs_C.mutable_layout());
    // ruy_rhs_C.set_data(kernel_data_C);
    // ruy_rhs_C.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create dst
    ruy::MakeSimpleLayout(1, output_shape.size[0], ruy::Order::kColMajor,
                              ruy_dst.mutable_layout());
    ruy_dst.set_data(output_data_R);

    ruy::Context* _ruy_context = new ruy::Context;
    ruy::MulParams<int32_t, int32_t> ruy_mul_params;

    ruy::Mul(ruy_lhs, ruy_rhs, ruy_mul_params, _ruy_context, &ruy_dst);

#if PRINT_VALUES || PRINT_MUL_OUTPUT
    cout << "[";
    for (int i = 0; i < output_shape.size[0]; i++)
        cout << ((int)output_data_R[i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    minus_one_vector(output_data, output_shape.flatsize);
    ret = LowPrecision::FullyConnected::Int4::MultiplyInt8SingleBatch(
                            input_pad_data, input_pad_shape,
                            filter_data, kernel_pad_shape,
                            output_data, output_shape);
    bool MI8I4Passed = true;
    for (int i = 0 ; i < output_shape.flatsize / 2 ; i++)
        MI8I4Passed &= (output_data_R[i] == output_data[i]);

    cout << "LowPrecision::FullyConnected::Int4::MultiplyInt8SingleBatch Return Status\t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    cout << "LowPrecision::FullyConnected::Int4::MultiplyInt8SingleBatch\t\t\t=> " << ((MI8I4Passed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (MI8I4Passed?"PASSED":"FAILED") << "\033[0m" << endl;
#if PRINT_VALUES || PRINT_MUL_OUTPUT
    cout << "[";
    for (int i = 0; i < output_shape.size[0]; i++)
        cout << ((int)output_data[i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    zero_vector(output_data_R_MB, output_shape_MB.flatsize);

    ruy::Matrix<int8_t> ruy_MB_lhs;
    ruy::Matrix<int8_t> ruy_MB_rhs;
    ruy::Matrix<int8_t> ruy_MB_rhs_C;
    ruy::Matrix<int32_t> ruy_MB_dst;

    // Create lhs
    ruy::MakeSimpleLayout(input_shape_MB.size[0], input_shape_MB.size[1], ruy::Order::kColMajor,
                              ruy_MB_lhs.mutable_layout());
    ruy_MB_lhs.set_data(input_data_MB);
    // Create rhs (Column Major)
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kColMajor,
                              ruy_MB_rhs.mutable_layout());
    ruy_MB_rhs.set_data(kernel_data);
    ruy_MB_rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create rhs (Row Major)
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kRowMajor,
                              ruy_MB_rhs_C.mutable_layout());
    ruy_MB_rhs_C.set_data(kernel_data_C);
    ruy_MB_rhs_C.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create dst
    ruy::MakeSimpleLayout(output_shape_MB.size[0], output_shape_MB.size[1], ruy::Order::kColMajor,
                              ruy_MB_dst.mutable_layout());
    ruy_MB_dst.set_data(output_data_R_MB);

    ruy::Context* _ruy_MB_context = new ruy::Context;
    ruy::MulParams<int32_t, int32_t> ruy_MB_mul_params;

    ruy::Mul(ruy_MB_lhs, ruy_MB_rhs, ruy_MB_mul_params, _ruy_MB_context, &ruy_MB_dst);

#if PRINT_VALUES || PRINT_MUL_OUTPUT || PRINT_MUL_MB_OUTPUT
    cout << "[";
    for (int j = 0; j < output_shape_MB.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < output_shape_MB.size[1]; i++)
            cout << ((int)output_data_R_MB[j * output_shape_MB.size[1] + i]) << ", ";
        cout << "],\n";
    }
    cout << "]";
    cout << endl;
#endif

    zero_vector(output_data_MB, output_shape_MB.flatsize);
    ret = LowPrecision::FullyConnected::Int4::MultiplyInt8MultiBatched(
                            input_pack_MB, input_shape_MB,
                            filter_data, kernel_shape,
                            output_data_MB, output_shape_MB);
    bool MI8I4MBPassed = true;
    for (int i = 0 ; i < output_shape_MB.flatsize / 2 ; i++)
        MI8I4MBPassed &= (output_data_R_MB[i] == output_data_MB[i]);
    cout << "LowPrecision::FullyConnected::Int4::MultiplyInt8-MB Return Status \t\t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    cout << "LowPrecision::FullyConnected::Int4::MultiplyInt8-MB \t\t\t\t=> " << ((MI8I4MBPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (MI8I4MBPassed?"PASSED":"FAILED") << "\033[0m" << endl;
#if PRINT_VALUES || PRINT_MUL_OUTPUT || PRINT_MUL_MB_OUTPUT
    cout << "[";
    for (int j = 0; j < output_shape_MB.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < output_shape_MB.size[1]; i++)
            cout << ((int)output_data_MB[j * output_shape_MB.size[1] + i]) << ", ";
        cout << "],\n";
    }
    cout << "]";
    cout << endl;
#endif
    
    deallocate(input_data);
    deallocate(input_data_MB);
    deallocate(input_pack_MB);
    deallocate(kernel_data);
    deallocate(kernel_data_C);
    deallocate(filter_data);
    deallocate(output_data);
    deallocate(output_data_MB);
    deallocate(output_data_R);
    deallocate(output_data_R_MB);
    cout << "LowPrecision::FullyConnected::Int4  Deallocation\t\t\t\t=> \033[1m\033[32mPASSED\033[0m" << endl;
}

void run_i8bin_tests(
    const int* _template,
    const int8_t* _answers,
    const int kernel_fill_mode,
    const int num_inputs,
    const int num_output,
    const int num_batch
    ){
    int _input_shape[1]       = { num_inputs },
        _input_shape_MB[2]    = { num_batch , num_inputs },
        _kernel_shape[2]      = { num_output, num_inputs },
        _filter_shape[2]      = { num_output, num_inputs / 8 },
        _output_shape[1]      = { num_output },
        _output_shape_MB[2]   = { num_batch , num_output };
    Shape input_shape         = get_shape(_input_shape,    1),
          input_shape_MB      = get_shape(_input_shape_MB, 2),
          kernel_shape        = get_shape(_kernel_shape,   2),
          filter_shape        = get_shape(_filter_shape,   2),
          output_shape        = get_shape(_output_shape,   1),
          output_shape_MB     = get_shape(_output_shape_MB,   2);
    
    int8_t*  input_data       = allocate<int8_t>(input_shape.flatsize);
    int8_t*  input_data_MB    = allocate<int8_t>(input_shape_MB.flatsize);
    int8_t*  input_pack_MB    = allocate<int8_t>(input_shape_MB.flatsize);
    int8_t*  kernel_data      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  kernel_data_C    = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  filter_data      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output_data      = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_MB   = allocate<int32_t>(output_shape_MB.flatsize);
    int32_t* output_data_R    = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_R_MB = allocate<int32_t>(output_shape_MB.flatsize);

    for (int i = 0; i < num_inputs; i++)
        input_data[i] = 1;
    
    for (int i = 0; i < num_batch; i++)
        for (int j = 0; j < num_inputs; j++)
            input_data_MB[i * num_inputs + j] = 1;
    
    if(kernel_fill_mode == 0){
        for (int i = 0; i < kernel_shape.size[0]; i++)
            for (int j = 0; j < kernel_shape.size[1]; j++)
                kernel_data[i * kernel_shape.size[1] + j]   = _template[j % num_inputs];

        for (int i = 0; i < kernel_shape.size[1]; i++)
            for (int j = 0; j < kernel_shape.size[0]; j++)
                kernel_data_C[i * kernel_shape.size[0] + j] = _template[i % num_inputs];
    }
    else if (kernel_fill_mode == 1){
        zero_vector(kernel_data, kernel_shape.flatsize);
        zero_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 2){
        one_vector(kernel_data, kernel_shape.flatsize);
        one_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 3){
        minus_one_vector(kernel_data, kernel_shape.flatsize);
        minus_one_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 4){
        two_vector(kernel_data, kernel_shape.flatsize);
        two_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 5){
        minus_two_vector(kernel_data, kernel_shape.flatsize);
        minus_two_vector(kernel_data_C, kernel_shape.flatsize);
    }

#if PRINT_VALUES
    cout << "Kernel = [" << endl;
    for (int i = 0; i < kernel_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < kernel_shape.size[1]; j++)
            cout << ((int)kernel_data[i * kernel_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "Kernel(0x) = [" << hex << endl;
    for (int i = 0; i < kernel_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < kernel_shape.size[1]; j++)
            cout << (((int)kernel_data[i * kernel_shape.size[1] + j]) & 0x01) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
#if PRINT_VALUES
    cout << "Kernel_C = [" << endl;
    for (int j = 0; j < kernel_shape.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < kernel_shape.size[1]; i++)
            cout << ((int)kernel_data_C[i * kernel_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "Kernel_C(0x) = [" << hex << endl;
    for (int j = 0; j < kernel_shape.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < kernel_shape.size[1]; i++)
            cout << (((int)kernel_data_C[i * kernel_shape.size[0] + j]) & 0x01) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    Status ret = LowPrecision::FullyConnected::Binary::QuantizeFilter(kernel_data, kernel_shape, filter_data, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::Binary::QuantizeFilter-RowMajor Return Status \t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 128 * 16){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4RMPassed &= (_answers[i] == filter_data[i]);
        cout << "LowPrecision::FullyConnected::Binary::QuantizeFilter-RowMajor    \t\t=> " << ((QFTI4RMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
        if(!QFTI4RMPassed){
            cout << "[" << endl;
            for (int i = 0; i < filter_shape.size[0]; i++){
                cout << "\t[";
                for (int j = 0; j < filter_shape.size[1]; j++)
                    cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
                cout << "]," << endl;
            }
            cout << "]";
            cout << endl;
        }
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((((uint8_t)filter_data[i * filter_shape.size[1] + j]) < 16)?("0"):("")) << (((int)filter_data[i * filter_shape.size[1] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    ret = LowPrecision::FullyConnected::Binary::QuantizeFilter(kernel_data_C, kernel_shape, filter_data, MemLayout::kColumnMajor);
    cout << "LowPrecision::FullyConnected::Binary::QuantizeFilter-ColumnMajor Return Status\t=> " << ((ret)?((ret == LowPrecision::Status::WrongMemLayout)?("\033[1m\033[33m"):("\033[1m\033[31m")):("\033[1m\033[32m")) << ((ret)?((ret == LowPrecision::Status::WrongMemLayout)?("NO_SUPPORT"):("FAILED")):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 128 * 16 && !ret){
        bool QFTI4CMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4CMPassed &= (_answers[i] == filter_data[i]);
        cout << "LowPrecision::FullyConnected::Binary::QuantizeFilter-ColumnMajor \t\t=> " << ((QFTI4CMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4CMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
        if(!QFTI4CMPassed){
            cout << "[" << endl;
            for (int i = 0; i < filter_shape.size[0]; i++){
                cout << "\t[";
                for (int j = 0; j < filter_shape.size[1]; j++)
                    cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
                cout << "]," << endl;
            }
            cout << "]";
            cout << endl;
        }
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((((uint8_t)filter_data[i * filter_shape.size[1] + j]) < 16)?("0"):("")) << (((int)filter_data[i * filter_shape.size[1] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif

    doLowPrecisionPack(input_data_MB, input_pack_MB, input_shape_MB.size[0], input_shape_MB.size[1]);

#if PRINT_VALUES || PRINT_ACTIVATIONS
    cout << "[" << endl;
    for (int i = 0; i < input_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape_MB.size[1]; j++)
            cout << ((int)input_data_MB[i * input_shape_MB.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES || PRINT_ACTIVATIONS
    cout << "[" << endl;
    for (int i = 0; i < input_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape_MB.size[1]; j++)
            cout << ((int)input_pack_MB[i * input_shape_MB.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < input_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape_MB.size[1]; j++)
            cout << (uint8_t)input_pack_MB[i * input_shape_MB.size[1] + j] << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    minus_one_vector(output_data_R, output_shape.flatsize);

    ruy::Matrix<int8_t> ruy_lhs;
    ruy::Matrix<int8_t> ruy_rhs;
    ruy::Matrix<int8_t> ruy_rhs_C;
    ruy::Matrix<int32_t> ruy_dst;

    // Create lhs
    ruy::MakeSimpleLayout(1, input_shape.size[0], ruy::Order::kColMajor,
                              ruy_lhs.mutable_layout());
    ruy_lhs.set_data(input_data);
    // Create rhs
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kColMajor,
                              ruy_rhs.mutable_layout());
    ruy_rhs.set_data(kernel_data);
    ruy_rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create rhs
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kRowMajor,
                              ruy_rhs_C.mutable_layout());
    ruy_rhs_C.set_data(kernel_data_C);
    ruy_rhs_C.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create dst
    ruy::MakeSimpleLayout(1, output_shape.size[0], ruy::Order::kColMajor,
                              ruy_dst.mutable_layout());
    ruy_dst.set_data(output_data_R);

    ruy::Context* _ruy_context = new ruy::Context;
    ruy::MulParams<int32_t, int32_t> ruy_mul_params;

    ruy::Mul(ruy_lhs, ruy_rhs, ruy_mul_params, _ruy_context, &ruy_dst);

#if PRINT_VALUES || PRINT_MUL_OUTPUT
    cout << "[";
    for (int i = 0; i < output_shape.size[0]; i++)
        cout << ((int)output_data_R[i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    minus_two_vector(output_data, output_shape.flatsize);
    ret = LowPrecision::FullyConnected::Binary::MultiplyInt8SingleBatch(
                            input_data, input_shape,
                            filter_data, kernel_shape,
                            output_data, output_shape
    );
    bool MI8I4Passed = true;
    for (int i = 0 ; i < output_shape.flatsize / 2 ; i++)
        MI8I4Passed &= (output_data_R[i] == output_data[i]);

    cout << "LowPrecision::FullyConnected::Binary::MultiplyInt8 Return Status \t\t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    cout << "LowPrecision::FullyConnected::Binary::MultiplyInt8 \t\t\t\t=> " << ((MI8I4Passed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (MI8I4Passed?"PASSED":"FAILED") << "\033[0m" << endl;
#if PRINT_VALUES || PRINT_MUL_OUTPUT
    cout << "[";
    for (int i = 0; i < output_shape.size[0]; i++)
        cout << ((int)output_data[i]) << ", ";
    cout << "]";
    cout << endl;
#endif
    zero_vector(output_data_R_MB, output_shape_MB.flatsize);

    ruy::Matrix<int8_t> ruy_MB_lhs;
    ruy::Matrix<int8_t> ruy_MB_rhs;
    ruy::Matrix<int8_t> ruy_MB_rhs_C;
    ruy::Matrix<int32_t> ruy_MB_dst;

    // Create lhs
    ruy::MakeSimpleLayout(input_shape_MB.size[0], input_shape_MB.size[1], ruy::Order::kColMajor,
                              ruy_MB_lhs.mutable_layout());
    ruy_MB_lhs.set_data(input_data_MB);
    // Create rhs (Column Major)
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kColMajor,
                              ruy_MB_rhs.mutable_layout());
    ruy_MB_rhs.set_data(kernel_data);
    ruy_MB_rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create rhs (Row Major)
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kRowMajor,
                              ruy_MB_rhs_C.mutable_layout());
    ruy_MB_rhs_C.set_data(kernel_data_C);
    ruy_MB_rhs_C.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create dst
    ruy::MakeSimpleLayout(output_shape_MB.size[0], output_shape_MB.size[1], ruy::Order::kColMajor,
                              ruy_MB_dst.mutable_layout());
    ruy_MB_dst.set_data(output_data_R_MB);

    ruy::Context* _ruy_MB_context = new ruy::Context;
    ruy::MulParams<int32_t, int32_t> ruy_MB_mul_params;

    ruy::Mul(ruy_MB_lhs, ruy_MB_rhs, ruy_MB_mul_params, _ruy_MB_context, &ruy_MB_dst);

#if PRINT_VALUES || PRINT_MUL_OUTPUT || PRINT_MUL_MB_OUTPUT
    cout << "[";
    for (int j = 0; j < output_shape_MB.size[0]; j++)
        for (int i = 0; i < output_shape_MB.size[1]; i++)
            cout << ((int)output_data_R_MB[j * output_shape_MB.size[1] + i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    zero_vector(output_data_MB, output_shape_MB.flatsize);
    ret = LowPrecision::FullyConnected::Binary::MultiplyInt8MultiBatched(
                            input_pack_MB, input_shape_MB,
                            filter_data, kernel_shape,
                            output_data_MB, output_shape_MB);
    bool MI8I4MBPassed = true;
    for (int i = 0 ; i < output_shape_MB.flatsize / 2 ; i++)
        MI8I4MBPassed &= (output_data_R_MB[i] == output_data_MB[i]);
    cout << "LowPrecision::FullyConnected::Binary::MultiplyInt8-MB Return Status \t\t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    cout << "LowPrecision::FullyConnected::Binary::MultiplyInt8-MB \t\t\t\t=> " << ((MI8I4MBPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (MI8I4MBPassed?"PASSED":"FAILED") << "\033[0m" << endl;
#if PRINT_VALUES || PRINT_MUL_OUTPUT || PRINT_MUL_MB_OUTPUT
    cout << "[";
    for (int j = 0; j < output_shape_MB.size[0]; j++)
        for (int i = 0; i < output_shape_MB.size[1]; i++)
            cout << ((int)output_data_MB[j * output_shape_MB.size[1] + i]) << ", ";
    cout << "]";
    cout << endl;
#endif
    deallocate(input_data);
    deallocate(input_data_MB);
    deallocate(input_pack_MB);
    deallocate(kernel_data);
    deallocate(kernel_data_C);
    deallocate(filter_data);
    deallocate(output_data);
    deallocate(output_data_MB);
    deallocate(output_data_R);
    deallocate(output_data_R_MB);
    cout << "LowPrecision::FullyConnected::Binary  Deallocation\t\t\t\t=> \033[1m\033[32mPASSED\033[0m" << endl;
}

void run_i8ter_tests(
    const int* _template,
    const int8_t* _answers,
    const int kernel_fill_mode,
    const int num_inputs,
    const int num_output,
    const int num_batch
    ){
    int _input_shape[1]       = { num_inputs },
        _input_shape_MB[2]    = { num_batch , num_inputs },
        _kernel_shape[2]      = { num_output, num_inputs },
        _filter_shape[2]      = { num_output, num_inputs / 4 },
        _output_shape[1]      = { num_output },
        _output_shape_MB[2]   = { num_batch , num_output };
    Shape input_shape         = get_shape(_input_shape,     1),
          input_shape_MB      = get_shape(_input_shape_MB,  2),
          kernel_shape        = get_shape(_kernel_shape,    2),
          filter_shape        = get_shape(_filter_shape,    2),
          output_shape        = get_shape(_output_shape,    1),
          output_shape_MB     = get_shape(_output_shape_MB, 2);
    
    int8_t*  input_data       = allocate<int8_t>(input_shape.flatsize);
    int8_t*  input_data_MB    = allocate<int8_t>(input_shape_MB.flatsize);
    int8_t*  input_pack_MB    = allocate<int8_t>(input_shape_MB.flatsize);
    int8_t*  kernel_data      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  kernel_data_C    = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  filter_data      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output_data      = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_MB   = allocate<int32_t>(output_shape_MB.flatsize);
    int32_t* output_data_R    = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_R_MB = allocate<int32_t>(output_shape_MB.flatsize);

    for (int i = 0; i < num_inputs; i++)
        input_data[i] = 1;
    
    for (int i = 0; i < num_batch; i++)
        for (int j = 0; j < num_inputs; j++)
            input_data_MB[i * num_inputs + j] = 1;
    
    if(kernel_fill_mode == 0){
        for (int i = 0; i < kernel_shape.size[0]; i++)
            for (int j = 0; j < kernel_shape.size[1]; j++)
                kernel_data[i * kernel_shape.size[1] + j]   = _template[j % num_inputs];

        for (int i = 0; i < kernel_shape.size[1]; i++)
            for (int j = 0; j < kernel_shape.size[0]; j++)
                kernel_data_C[i * kernel_shape.size[0] + j] = _template[i % num_inputs];
    }
    else if (kernel_fill_mode == 1){
        zero_vector(kernel_data, kernel_shape.flatsize);
        zero_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 2){
        one_vector(kernel_data, kernel_shape.flatsize);
        one_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 3){
        minus_one_vector(kernel_data, kernel_shape.flatsize);
        minus_one_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 4){
        two_vector(kernel_data, kernel_shape.flatsize);
        two_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 5){
        minus_two_vector(kernel_data, kernel_shape.flatsize);
        minus_two_vector(kernel_data_C, kernel_shape.flatsize);
    }

#if PRINT_VALUES
    cout << "Kernel = [" << endl;
    for (int i = 0; i < kernel_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < kernel_shape.size[1]; j++)
            cout << ((int)kernel_data[i * kernel_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "Kernel(0x) = [" << hex << endl;
    for (int i = 0; i < kernel_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < kernel_shape.size[1]; j++)
            cout << (((int)kernel_data[i * kernel_shape.size[1] + j]) & 0x01) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
#if PRINT_VALUES
    cout << "Kernel_C = [" << endl;
    for (int j = 0; j < kernel_shape.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < kernel_shape.size[1]; i++)
            cout << ((int)kernel_data_C[i * kernel_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "Kernel_C(0x) = [" << hex << endl;
    for (int j = 0; j < kernel_shape.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < kernel_shape.size[1]; i++)
            cout << (((int)kernel_data_C[i * kernel_shape.size[0] + j]) & 0x01) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    Status ret = LowPrecision::FullyConnected::Ternary::QuantizeFilter(kernel_data, kernel_shape, filter_data, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::Ternary::QuantizeFilter-RowMajor Return Status \t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 64 * 16){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4RMPassed &= (_answers[i] == filter_data[i]);
        cout << "LowPrecision::FullyConnected::Ternary::QuantizeFilter-RowMajor    \t\t=> " << ((QFTI4RMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
        if(!QFTI4RMPassed){
            cout << "[" << endl;
            for (int i = 0; i < filter_shape.size[0]; i++){
                cout << "\t[";
                for (int j = 0; j < filter_shape.size[1]; j++)
                    cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
                cout << "]," << endl;
            }
            cout << "]";
            cout << endl;
        }
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((((uint8_t)filter_data[i * filter_shape.size[1] + j]) < 16)?("0"):("")) << (((int)filter_data[i * filter_shape.size[1] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    ret = LowPrecision::FullyConnected::Ternary::QuantizeFilter(kernel_data_C, kernel_shape, filter_data, MemLayout::kColumnMajor);
    cout << "LowPrecision::FullyConnected::Ternary::QuantizeFilter-ColumnMajor Return Status\t=> " << ((ret)?((ret == LowPrecision::Status::WrongMemLayout)?("\033[1m\033[33m"):("\033[1m\033[31m")):("\033[1m\033[32m")) << ((ret)?((ret == LowPrecision::Status::WrongMemLayout)?("NO_SUPPORT"):("FAILED")):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 128 * 16 && !ret){
        bool QFTI4CMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4CMPassed &= (_answers[i] == filter_data[i]);
        cout << "LowPrecision::FullyConnected::Ternary::QuantizeFilter-ColumnMajor \t\t=> " << ((QFTI4CMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4CMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
        if(!QFTI4CMPassed){
            cout << "[" << endl;
            for (int i = 0; i < filter_shape.size[0]; i++){
                cout << "\t[";
                for (int j = 0; j < filter_shape.size[1]; j++)
                    cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
                cout << "]," << endl;
            }
            cout << "]";
            cout << endl;
        }
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((((uint8_t)filter_data[i * filter_shape.size[1] + j]) < 16)?("0"):("")) << (((int)filter_data[i * filter_shape.size[1] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif

    doLowPrecisionPack(input_data_MB, input_pack_MB, input_shape_MB.size[0], input_shape_MB.size[1]);

#if PRINT_VALUES || PRINT_ACTIVATIONS
    cout << "[" << endl;
    for (int i = 0; i < input_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape_MB.size[1]; j++)
            cout << ((int)input_data_MB[i * input_shape_MB.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES || PRINT_ACTIVATIONS
    cout << "[" << endl;
    for (int i = 0; i < input_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape_MB.size[1]; j++)
            cout << ((int)input_pack_MB[i * input_shape_MB.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < input_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape_MB.size[1]; j++)
            cout << (uint8_t)input_pack_MB[i * input_shape_MB.size[1] + j] << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    minus_one_vector(output_data_R, output_shape.flatsize);

    ruy::Matrix<int8_t> ruy_lhs;
    ruy::Matrix<int8_t> ruy_rhs;
    ruy::Matrix<int8_t> ruy_rhs_C;
    ruy::Matrix<int32_t> ruy_dst;

    // Create lhs
    ruy::MakeSimpleLayout(1, input_shape.size[0], ruy::Order::kColMajor,
                              ruy_lhs.mutable_layout());
    ruy_lhs.set_data(input_data);
    // Create rhs
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kColMajor,
                              ruy_rhs.mutable_layout());
    ruy_rhs.set_data(kernel_data);
    ruy_rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create rhs
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kRowMajor,
                              ruy_rhs_C.mutable_layout());
    ruy_rhs_C.set_data(kernel_data_C);
    ruy_rhs_C.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create dst
    ruy::MakeSimpleLayout(1, output_shape.size[0], ruy::Order::kColMajor,
                              ruy_dst.mutable_layout());
    ruy_dst.set_data(output_data_R);

    ruy::Context* _ruy_context = new ruy::Context;
    ruy::MulParams<int32_t, int32_t> ruy_mul_params;

    ruy::Mul(ruy_lhs, ruy_rhs, ruy_mul_params, _ruy_context, &ruy_dst);

#if PRINT_VALUES || PRINT_MUL_OUTPUT
    cout << "[";
    for (int i = 0; i < output_shape.size[0]; i++)
        cout << ((int)output_data_R[i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    minus_two_vector(output_data, output_shape.flatsize);
    ret = LowPrecision::FullyConnected::Ternary::MultiplyInt8SingleBatch(
                            input_data, input_shape,
                            filter_data, kernel_shape,
                            output_data, output_shape
    );
    bool MI8I4Passed = true;
    for (int i = 0 ; i < output_shape.flatsize / 2 ; i++)
        MI8I4Passed &= (output_data_R[i] == output_data[i]);

    cout << "LowPrecision::FullyConnected::Ternary::MultiplyInt8 Return Status \t\t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    cout << "LowPrecision::FullyConnected::Ternary::MultiplyInt8 \t\t\t\t=> " << ((MI8I4Passed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (MI8I4Passed?"PASSED":"FAILED") << "\033[0m" << endl;
#if PRINT_VALUES || PRINT_MUL_OUTPUT
    cout << "[";
    for (int i = 0; i < output_shape.size[0]; i++)
        cout << ((int)output_data[i]) << ", ";
    cout << "]";
    cout << endl;
#endif
    zero_vector(output_data_R_MB, output_shape_MB.flatsize);

    ruy::Matrix<int8_t> ruy_MB_lhs;
    ruy::Matrix<int8_t> ruy_MB_rhs;
    ruy::Matrix<int8_t> ruy_MB_rhs_C;
    ruy::Matrix<int32_t> ruy_MB_dst;

    // Create lhs
    ruy::MakeSimpleLayout(input_shape_MB.size[0], input_shape_MB.size[1], ruy::Order::kColMajor,
                              ruy_MB_lhs.mutable_layout());
    ruy_MB_lhs.set_data(input_data_MB);
    // Create rhs (Column Major)
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kColMajor,
                              ruy_MB_rhs.mutable_layout());
    ruy_MB_rhs.set_data(kernel_data);
    ruy_MB_rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create rhs (Row Major)
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kRowMajor,
                              ruy_MB_rhs_C.mutable_layout());
    ruy_MB_rhs_C.set_data(kernel_data_C);
    ruy_MB_rhs_C.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create dst
    ruy::MakeSimpleLayout(output_shape_MB.size[0], output_shape_MB.size[1], ruy::Order::kColMajor,
                              ruy_MB_dst.mutable_layout());
    ruy_MB_dst.set_data(output_data_R_MB);

    ruy::Context* _ruy_MB_context = new ruy::Context;
    ruy::MulParams<int32_t, int32_t> ruy_MB_mul_params;

    ruy::Mul(ruy_MB_lhs, ruy_MB_rhs, ruy_MB_mul_params, _ruy_MB_context, &ruy_MB_dst);

#if PRINT_VALUES || PRINT_MUL_OUTPUT || PRINT_MUL_MB_OUTPUT
    cout << "[";
    for (int j = 0; j < output_shape_MB.size[0]; j++)
        for (int i = 0; i < output_shape_MB.size[1]; i++)
            cout << ((int)output_data_R_MB[j * output_shape_MB.size[1] + i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    zero_vector(output_data_MB, output_shape_MB.flatsize);
    ret = LowPrecision::FullyConnected::Ternary::MultiplyInt8MultiBatched(input_pack_MB, input_shape_MB,
                            filter_data, kernel_shape,
                            output_data_MB, output_shape_MB);
    bool MI8I4MBPassed = true;
    for (int i = 0 ; i < output_shape_MB.flatsize / 2 ; i++)
        MI8I4MBPassed &= (output_data_R_MB[i] == output_data_MB[i]);
    cout << "LowPrecision::FullyConnected::Ternary::MultiplyInt8MultiBatched Return Status\t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    cout << "LowPrecision::FullyConnected::Ternary::MultiplyInt8MultiBatched\t\t\t=> " << ((MI8I4MBPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (MI8I4MBPassed?"PASSED":"FAILED") << "\033[0m" << endl;
#if PRINT_VALUES || PRINT_MUL_OUTPUT || PRINT_MUL_MB_OUTPUT
    cout << "[";
    for (int j = 0; j < output_shape_MB.size[0]; j++)
        for (int i = 0; i < output_shape_MB.size[1]; i++)
            cout << ((int)output_data_MB[j * output_shape_MB.size[1] + i]) << ", ";
    cout << "]";
    cout << endl;
#endif
    deallocate(input_data);
    deallocate(input_data_MB);
    deallocate(input_pack_MB);
    deallocate(kernel_data);
    deallocate(kernel_data_C);
    deallocate(filter_data);
    deallocate(output_data);
    deallocate(output_data_MB);
    deallocate(output_data_R);
    deallocate(output_data_R_MB);
    cout << "LowPrecision::FullyConnected::Ternary  Deallocation\t\t\t\t=> \033[1m\033[32mPASSED\033[0m" << endl;
}

void run_i8qua_tests(
    const int* _template,
    const int8_t* _answers,
    const int kernel_fill_mode,
    const int num_inputs,
    const int num_output,
    const int num_batch
    ){
    int _input_shape[1]       = { num_inputs },
        _input_shape_MB[2]    = { num_batch , num_inputs },
        _kernel_shape[2]      = { num_output, num_inputs },
        _filter_shape[2]      = { num_output, num_inputs / 8 },
        _output_shape[1]      = { num_output },
        _output_shape_MB[2]   = { num_batch , num_output };
    Shape input_shape         = get_shape(_input_shape,    1),
          input_shape_MB      = get_shape(_input_shape_MB, 2),
          kernel_shape        = get_shape(_kernel_shape,   2),
          filter_shape        = get_shape(_filter_shape,   2),
          output_shape        = get_shape(_output_shape,   1),
          output_shape_MB     = get_shape(_output_shape_MB,   2);
    
    int8_t*  input_data       = allocate<int8_t>(input_shape.flatsize);
    int8_t*  input_data_MB    = allocate<int8_t>(input_shape_MB.flatsize);
    int8_t*  input_pack_MB    = allocate<int8_t>(input_shape_MB.flatsize);
    int8_t*  kernel_data      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  kernel_data_C    = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  filter_data      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output_data      = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_MB   = allocate<int32_t>(output_shape_MB.flatsize);
    int32_t* output_data_R    = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_R_MB = allocate<int32_t>(output_shape_MB.flatsize);

    for (int i = 0; i < num_inputs; i++)
        input_data[i] = 1;
    
    for (int i = 0; i < num_batch; i++)
        for (int j = 0; j < num_inputs; j++)
            input_data_MB[i * num_inputs + j] = 1;
    
    if(kernel_fill_mode == 0){
        for (int i = 0; i < kernel_shape.size[0]; i++)
            for (int j = 0; j < kernel_shape.size[1]; j++)
                kernel_data[i * kernel_shape.size[1] + j]   = _template[j % num_inputs];

        for (int i = 0; i < kernel_shape.size[1]; i++)
            for (int j = 0; j < kernel_shape.size[0]; j++)
                kernel_data_C[i * kernel_shape.size[0] + j] = _template[i % num_inputs];
    }
    else if (kernel_fill_mode == 1){
        zero_vector(kernel_data, kernel_shape.flatsize);
        zero_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 2){
        one_vector(kernel_data, kernel_shape.flatsize);
        one_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 3){
        minus_one_vector(kernel_data, kernel_shape.flatsize);
        minus_one_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 4){
        two_vector(kernel_data, kernel_shape.flatsize);
        two_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 5){
        minus_two_vector(kernel_data, kernel_shape.flatsize);
        minus_two_vector(kernel_data_C, kernel_shape.flatsize);
    }

#if PRINT_VALUES
    cout << "Kernel = [" << endl;
    for (int i = 0; i < kernel_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < kernel_shape.size[1]; j++)
            cout << ((int)kernel_data[i * kernel_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "Kernel(0x) = [" << hex << endl;
    for (int i = 0; i < kernel_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < kernel_shape.size[1]; j++)
            cout << (((int)kernel_data[i * kernel_shape.size[1] + j]) & 0x01) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
#if PRINT_VALUES
    cout << "Kernel_C = [" << endl;
    for (int j = 0; j < kernel_shape.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < kernel_shape.size[1]; i++)
            cout << ((int)kernel_data_C[i * kernel_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "Kernel_C(0x) = [" << hex << endl;
    for (int j = 0; j < kernel_shape.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < kernel_shape.size[1]; i++)
            cout << (((int)kernel_data_C[i * kernel_shape.size[0] + j]) & 0x01) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    Status ret = LowPrecision::FullyConnected::Quaternary::QuantizeFilter(kernel_data, kernel_shape, filter_data, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::Quaternary::QuantizeFilter-RowMajor Return Status\t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 128 * 16){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4RMPassed &= (_answers[i] == filter_data[i]);
        cout << "LowPrecision::FullyConnected::Quaternary::QuantizeFilter-RowMajor    \t\t=> " << ((QFTI4RMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
        if(!QFTI4RMPassed){
            cout << "[" << endl;
            for (int i = 0; i < filter_shape.size[0]; i++){
                cout << "\t[";
                for (int j = 0; j < filter_shape.size[1]; j++)
                    cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
                cout << "]," << endl;
            }
            cout << "]";
            cout << endl;
        }
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((((uint8_t)filter_data[i * filter_shape.size[1] + j]) < 16)?("0"):("")) << (((int)filter_data[i * filter_shape.size[1] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    ret = LowPrecision::FullyConnected::Quaternary::QuantizeFilter(kernel_data_C, kernel_shape, filter_data, MemLayout::kColumnMajor);
    cout << "LowPrecision::FullyConnected::Quaternary::QuantizeFilter-ColumnMajor Return Sta\t=> " << ((ret)?((ret == LowPrecision::Status::WrongMemLayout)?("\033[1m\033[33m"):("\033[1m\033[31m")):("\033[1m\033[32m")) << ((ret)?((ret == LowPrecision::Status::WrongMemLayout)?("NO_SUPPORT"):("FAILED")):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 128 * 16 && !ret){
        bool QFTI4CMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4CMPassed &= (_answers[i] == filter_data[i]);
        cout << "LowPrecision::FullyConnected::Quaternary::QuantizeFilter-ColumnMajor \t\t=> " << ((QFTI4CMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4CMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
        if(!QFTI4CMPassed){
            cout << "[" << endl;
            for (int i = 0; i < filter_shape.size[0]; i++){
                cout << "\t[";
                for (int j = 0; j < filter_shape.size[1]; j++)
                    cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
                cout << "]," << endl;
            }
            cout << "]";
            cout << endl;
        }
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((((uint8_t)filter_data[i * filter_shape.size[1] + j]) < 16)?("0"):("")) << (((int)filter_data[i * filter_shape.size[1] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif

    doLowPrecisionPack(input_data_MB, input_pack_MB, input_shape_MB.size[0], input_shape_MB.size[1]);

#if PRINT_VALUES || PRINT_ACTIVATIONS
    cout << "[" << endl;
    for (int i = 0; i < input_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape_MB.size[1]; j++)
            cout << ((int)input_data_MB[i * input_shape_MB.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES || PRINT_ACTIVATIONS
    cout << "[" << endl;
    for (int i = 0; i < input_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape_MB.size[1]; j++)
            cout << ((int)input_pack_MB[i * input_shape_MB.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < input_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape_MB.size[1]; j++)
            cout << (uint8_t)input_pack_MB[i * input_shape_MB.size[1] + j] << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    minus_one_vector(output_data_R, output_shape.flatsize);

    ruy::Matrix<int8_t> ruy_lhs;
    ruy::Matrix<int8_t> ruy_rhs;
    ruy::Matrix<int8_t> ruy_rhs_C;
    ruy::Matrix<int32_t> ruy_dst;

    // Create lhs
    ruy::MakeSimpleLayout(1, input_shape.size[0], ruy::Order::kColMajor,
                              ruy_lhs.mutable_layout());
    ruy_lhs.set_data(input_data);
    // Create rhs
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kColMajor,
                              ruy_rhs.mutable_layout());
    ruy_rhs.set_data(kernel_data);
    ruy_rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create rhs
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kRowMajor,
                              ruy_rhs_C.mutable_layout());
    ruy_rhs_C.set_data(kernel_data_C);
    ruy_rhs_C.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create dst
    ruy::MakeSimpleLayout(1, output_shape.size[0], ruy::Order::kColMajor,
                              ruy_dst.mutable_layout());
    ruy_dst.set_data(output_data_R);

    ruy::Context* _ruy_context = new ruy::Context;
    ruy::MulParams<int32_t, int32_t> ruy_mul_params;

    ruy::Mul(ruy_lhs, ruy_rhs, ruy_mul_params, _ruy_context, &ruy_dst);

#if PRINT_VALUES || PRINT_MUL_OUTPUT
    cout << "[";
    for (int i = 0; i < output_shape.size[0]; i++)
        cout << ((int)output_data_R[i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    minus_two_vector(output_data, output_shape.flatsize);
    ret = LowPrecision::FullyConnected::Quaternary::MultiplyInt8SingleBatch(
                            input_data, input_shape,
                            filter_data, kernel_shape,
                            output_data, output_shape
    );
    bool MI8I4Passed = true;
    for (int i = 0 ; i < output_shape.flatsize / 2 ; i++)
        MI8I4Passed &= (output_data_R[i] == output_data[i]);

    cout << "LowPrecision::FullyConnected::Quaternary::MultiplyInt8 Return Status \t\t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    cout << "LowPrecision::FullyConnected::Quaternary::MultiplyInt8 \t\t\t\t=> " << ((MI8I4Passed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (MI8I4Passed?"PASSED":"FAILED") << "\033[0m" << endl;
#if PRINT_VALUES || PRINT_MUL_OUTPUT
    cout << "[";
    for (int i = 0; i < output_shape.size[0]; i++)
        cout << ((int)output_data[i]) << ", ";
    cout << "]";
    cout << endl;
#endif
//     zero_vector(output_data_R_MB, output_shape_MB.flatsize);
//
//     ruy::Matrix<int8_t> ruy_MB_lhs;
//     ruy::Matrix<int8_t> ruy_MB_rhs;
//     ruy::Matrix<int8_t> ruy_MB_rhs_C;
//     ruy::Matrix<int32_t> ruy_MB_dst;
//
//     // Create lhs
//     ruy::MakeSimpleLayout(input_shape_MB.size[0], input_shape_MB.size[1], ruy::Order::kColMajor,
//                               ruy_MB_lhs.mutable_layout());
//     ruy_MB_lhs.set_data(input_data_MB);
//     // Create rhs (Column Major)
//     ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kColMajor,
//                               ruy_MB_rhs.mutable_layout());
//     ruy_MB_rhs.set_data(kernel_data);
//     ruy_MB_rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
//     // Create rhs (Row Major)
//     ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kRowMajor,
//                               ruy_MB_rhs_C.mutable_layout());
//     ruy_MB_rhs_C.set_data(kernel_data_C);
//     ruy_MB_rhs_C.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
//     // Create dst
//     ruy::MakeSimpleLayout(output_shape_MB.size[0], output_shape_MB.size[1], ruy::Order::kColMajor,
//                               ruy_MB_dst.mutable_layout());
//     ruy_MB_dst.set_data(output_data_R_MB);
//
//     ruy::Context* _ruy_MB_context = new ruy::Context;
//     ruy::MulParams<int32_t, int32_t> ruy_MB_mul_params;
//
//     ruy::Mul(ruy_MB_lhs, ruy_MB_rhs, ruy_MB_mul_params, _ruy_MB_context, &ruy_MB_dst);
//
// #if PRINT_VALUES || PRINT_MUL_OUTPUT || PRINT_MUL_MB_OUTPUT
//     cout << "[";
//     for (int j = 0; j < output_shape_MB.size[0]; j++)
//         for (int i = 0; i < output_shape_MB.size[1]; i++)
//             cout << ((int)output_data_R_MB[j * output_shape_MB.size[1] + i]) << ", ";
//     cout << "]";
//     cout << endl;
// #endif
//
//     zero_vector(output_data_MB, output_shape_MB.flatsize);
//     ret = LowPrecision::FullyConnected::Quaternary::MultiplyInt8(input_pack_MB, input_shape_MB,
//                             filter_data, kernel_shape,
//                             output_data_MB, output_shape_MB);
//     bool MI8I4MBPassed = true;
//     for (int i = 0 ; i < output_shape_MB.flatsize / 2 ; i++)
//         MI8I4MBPassed &= (output_data_R_MB[i] == output_data_MB[i]);
//     cout << "LowPrecision::FullyConnected::Quaternary::MultiplyInt8-MB Return Status \t\t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
//     cout << "LowPrecision::FullyConnected::Quaternary::MultiplyInt8-MB \t\t\t\t=> " << ((MI8I4MBPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (MI8I4MBPassed?"PASSED":"FAILED") << "\033[0m" << endl;
// #if PRINT_VALUES || PRINT_MUL_OUTPUT || PRINT_MUL_MB_OUTPUT
//     cout << "[";
//     for (int j = 0; j < output_shape_MB.size[0]; j++)
//         for (int i = 0; i < output_shape_MB.size[1]; i++)
//             cout << ((int)output_data_MB[j * output_shape_MB.size[1] + i]) << ", ";
//     cout << "]";
//     cout << endl;
// #endif
    deallocate(input_data);
    deallocate(input_data_MB);
    deallocate(input_pack_MB);
    deallocate(kernel_data);
    deallocate(kernel_data_C);
    deallocate(filter_data);
    deallocate(output_data);
    deallocate(output_data_MB);
    deallocate(output_data_R);
    deallocate(output_data_R_MB);
    cout << "LowPrecision::FullyConnected::Quaternary  Deallocation\t\t\t\t=> \033[1m\033[32mPASSED\033[0m" << endl;
}

void run_i4i8_tests(
    const int* _template,
    const int8_t* _answers,
    const int kernel_fill_mode,
    const int num_inputs,
    const int num_output,
    const int num_batch
    ){
    int _input_shape[1]         = { num_inputs },
        _activation_shape[1]    = { num_inputs / 2 },
        _input_shape_MB[2]      = { num_batch , num_inputs },
        _activation_MB_shape[2] = { num_batch , num_inputs / 2 },
        _kernel_shape[2]        = { num_output, num_inputs },
        _filter_shape[2]        = { num_output, num_inputs },
        _output_shape[1]        = { num_output },
        _output_shape_MB[2]     = { num_batch , num_output };
    Shape input_shape           = get_shape(_input_shape,         1),
          activation_shape      = get_shape(_activation_shape,    1),
          input_shape_MB        = get_shape(_input_shape_MB,      2),
          activation_shape_MB   = get_shape(_activation_MB_shape, 2),
          kernel_shape          = get_shape(_kernel_shape,        2),
          filter_shape          = get_shape(_filter_shape,        2),
          output_shape          = get_shape(_output_shape,        1),
          output_shape_MB       = get_shape(_output_shape_MB,     2);
    
    int8_t*  input_data         = allocate<int8_t>(input_shape.flatsize);
    int8_t*  activation_data    = allocate<int8_t>(activation_shape.flatsize);
    int8_t*  input_data_MB      = allocate<int8_t>(input_shape_MB.flatsize);
    int8_t*  activation_data_MB = allocate<int8_t>(input_shape_MB.flatsize);
    int8_t*  kernel_data        = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  kernel_data_C      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  filter_data        = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output_data        = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_MB     = allocate<int32_t>(output_shape_MB.flatsize);
    int32_t* output_data_R      = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_R_MB   = allocate<int32_t>(output_shape_MB.flatsize);

    one_vector(input_data, num_inputs);
    // half_one_half_zero_vector(input_data, num_inputs);
    
    for (int i = 0; i < num_batch; i++)
        for (int j = 0; j < num_inputs; j++)
            input_data_MB[i * num_inputs + j] = 1;
    
    if(kernel_fill_mode == 0){
        for (int i = 0; i < kernel_shape.size[0]; i++)
            for (int j = 0; j < kernel_shape.size[1]; j++)
                kernel_data[i * kernel_shape.size[1] + j]   = _template[j % 32];

        for (int i = 0; i < kernel_shape.size[1]; i++)
            for (int j = 0; j < kernel_shape.size[0]; j++)
                kernel_data_C[i * kernel_shape.size[0] + j] = _template[i % 32];
    }
    else if (kernel_fill_mode == 1){
        zero_vector(kernel_data, kernel_shape.flatsize);
        zero_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 2){
        one_vector(kernel_data, kernel_shape.flatsize);
        one_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 3){
        minus_one_vector(kernel_data, kernel_shape.flatsize);
        minus_one_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 4){
        two_vector(kernel_data, kernel_shape.flatsize);
        two_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 5){
        minus_two_vector(kernel_data, kernel_shape.flatsize);
        minus_two_vector(kernel_data_C, kernel_shape.flatsize);
    }

    Status i_ret = LowPrecision::FullyConnected::Int4InputsInt8Weights::QuantizeInput(input_data, input_shape, activation_data, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::Int4InputsInt8Weights::QuantizeInput-RowMajor Return Status\t=> " << ((i_ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((i_ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (activation_shape.flatsize <= 512){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < activation_shape.flatsize ; i++)
            QFTI4RMPassed &= (activation_data[i] == 17);
        cout << "LowPrecision::FullyConnected::Int4InputsInt8Weights::QuantizeInput-RowMajor\t\t\t=> " << ((QFTI4RMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < 1; i++){
        cout << "\t[";
        for (int j = 0; j < activation_shape.size[0]; j++)
            cout << ((int)activation_data[i * activation_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < 1; i++){
        cout << "\t[";
        for (int j = 0; j < activation_shape.size[0]; j++)
            cout << ((((uint8_t)activation_data[i * activation_shape.size[0] + j]) < 16)?("0"):("")) << (((int)activation_data[i * activation_shape.size[0] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif

    i_ret = LowPrecision::FullyConnected::Int4InputsInt8Weights::QuantizeInput(input_data_MB, input_shape_MB, activation_data_MB, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::Int4InputsInt8Weights::QuantizeInput-MB-RowMajor Return Status\t=> " << ((i_ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((i_ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (activation_shape_MB.flatsize <= 32 * 4 && !i_ret){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < activation_shape_MB.flatsize ; i++){
            if (activation_data_MB[i] != 17)
                cout << i << ": " << activation_data_MB[i] << " -> " << 17 << endl;
            QFTI4RMPassed &= (activation_data_MB[i] == 17);
        }
        cout << "LowPrecision::FullyConnected::Int4InputsInt8Weights::QuantizeInput-MB-RowMajor\t\t\t=> " << ((QFTI4RMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_VALUES || PRINT_ACTIVATIONS
    cout << LowPrecision::get_shape_string(input_shape_MB) << ", " << input_shape_MB.flatsize << endl;
    cout << "[" << endl;
    for (int i = 0; i < input_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape_MB.size[1]; j++)
            cout << ((int)activation_data_MB[i * input_shape_MB.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < activation_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < activation_shape_MB.size[1]; j++)
            cout << ((((uint8_t)activation_data_MB[i * activation_shape_MB.size[0] + j]) < 16)?("0"):("")) << (((int)activation_data_MB[i * activation_shape_MB.size[0] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif

#if PRINT_VALUES
    cout << "Kernel = [" << endl;
    for (int i = 0; i < kernel_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < kernel_shape.size[1]; j++)
            cout << ((int)kernel_data[i * kernel_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "Kernel(0x) = [" << hex << endl;
    for (int i = 0; i < kernel_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < kernel_shape.size[1]; j++)
            cout << (((int)kernel_data[i * kernel_shape.size[1] + j]) & 0x0f) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
#if PRINT_VALUES
    cout << "Kernel_C = [" << endl;
    for (int j = 0; j < kernel_shape.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < kernel_shape.size[1]; i++)
            cout << ((int)kernel_data_C[i * kernel_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "Kernel_C(0x) = [" << hex << endl;
    for (int j = 0; j < kernel_shape.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < kernel_shape.size[1]; i++)
            cout << (((int)kernel_data_C[i * kernel_shape.size[0] + j]) & 0x0f) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    Status ret = LowPrecision::FullyConnected::Int4InputsInt8Weights::QuantizeFilter(kernel_data, kernel_shape, filter_data, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::Int4InputsInt8Weights::QuantizeFilter-RowMajor Return Status \t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 512){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4RMPassed &= (_answers[i] == filter_data[i]);
        cout << "LowPrecision::FullyConnected::Int4InputsInt8Weights::QuantizeFilter-RowMajor    \t\t\t=> " << ((QFTI4RMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((((uint8_t)filter_data[i * filter_shape.size[1] + j]) < 16)?("0"):("")) << (((int)filter_data[i * filter_shape.size[1] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    ret = LowPrecision::FullyConnected::Int4InputsInt8Weights::QuantizeFilter(kernel_data_C, kernel_shape, filter_data, MemLayout::kColumnMajor);
    cout << "LowPrecision::FullyConnected::Int4InputsInt8Weights::QuantizeFilter-ColumnMajor Return Status\t=> " << ((ret)?((ret == LowPrecision::Status::WrongMemLayout)?("\033[1m\033[33m"):("\033[1m\033[31m")):("\033[1m\033[32m")) << ((ret)?((ret == LowPrecision::Status::WrongMemLayout)?("NO_SUPPORT"):("FAILED")):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 512 && !ret){
        bool QFTI4CMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4CMPassed &= (_answers[i] == filter_data[i]);
        cout << "LowPrecision::FullyConnected::Int4InputsInt8Weights::QuantizeFilter-ColumnMajor \t\t\t=> " << ((QFTI4CMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4CMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((((uint8_t)filter_data[i * filter_shape.size[1] + j]) < 16)?("0"):("")) << (((int)filter_data[i * filter_shape.size[1] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    minus_one_vector(output_data_R, output_shape.flatsize);

    ruy::Matrix<int8_t> ruy_lhs;
    ruy::Matrix<int8_t> ruy_rhs;
    ruy::Matrix<int8_t> ruy_rhs_C;
    ruy::Matrix<int32_t> ruy_dst;

    // Create lhs
    ruy::MakeSimpleLayout(1, input_shape.size[0], ruy::Order::kColMajor,
                              ruy_lhs.mutable_layout());
    ruy_lhs.set_data(input_data);
    // Create rhs
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kColMajor,
                              ruy_rhs.mutable_layout());
    ruy_rhs.set_data(kernel_data);
    ruy_rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create rhs
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kRowMajor,
                              ruy_rhs_C.mutable_layout());
    ruy_rhs_C.set_data(kernel_data_C);
    ruy_rhs_C.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create dst
    ruy::MakeSimpleLayout(1, output_shape.size[0], ruy::Order::kColMajor,
                              ruy_dst.mutable_layout());
    ruy_dst.set_data(output_data_R);

    ruy::Context* _ruy_context = new ruy::Context;
    ruy::MulParams<int32_t, int32_t> ruy_mul_params;

    ruy::Mul(ruy_lhs, ruy_rhs, ruy_mul_params, _ruy_context, &ruy_dst);

#if PRINT_VALUES || PRINT_MUL_OUTPUT
    cout << "[";
    for (int i = 0; i < output_shape.size[0]; i++)
        cout << ((int)output_data_R[i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    minus_one_vector(output_data, output_shape.flatsize);
    ret = LowPrecision::FullyConnected::Int4InputsInt8Weights::MultiplyInt8SingleBatch(
                            activation_data, input_shape,
                            filter_data, kernel_shape,
                            output_data, output_shape);
    bool MI8I4Passed = true;
    for (int i = 0 ; i < output_shape.flatsize / 2 ; i++)
        MI8I4Passed &= (output_data_R[i] == output_data[i]);

    cout << "LowPrecision::FullyConnected::Int4InputsInt8Weights::MultiplyInt8SingleBatch Return Status\t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    cout << "LowPrecision::FullyConnected::Int4InputsInt8Weights::MultiplyInt8SingleBatch\t\t\t=> " << ((MI8I4Passed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (MI8I4Passed?"PASSED":"FAILED") << "\033[0m" << endl;
#if PRINT_VALUES || PRINT_MUL_OUTPUT
    cout << "[";
    for (int i = 0; i < output_shape.size[0]; i++)
        cout << ((int)output_data[i]) << ", ";
    cout << "]";
    cout << endl;
#endif

#if PRINT_VALUES || PRINT_ACTIVATIONS
    cout << "[" << endl;
    for (int i = 0; i < input_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape_MB.size[1]; j++)
            cout << ((int)input_data_MB[i * input_shape_MB.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES || PRINT_ACTIVATIONS
    cout << "[" << endl;
    for (int i = 0; i < input_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape_MB.size[1]; j++)
            cout << ((int)activation_data_MB[i * input_shape_MB.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < input_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape_MB.size[1]; j++)
            cout << (uint8_t)activation_data_MB[i * input_shape_MB.size[1] + j] << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    zero_vector(output_data_R_MB, output_shape_MB.flatsize);

    ruy::Matrix<int8_t> ruy_MB_lhs;
    ruy::Matrix<int8_t> ruy_MB_rhs;
    ruy::Matrix<int8_t> ruy_MB_rhs_C;
    ruy::Matrix<int32_t> ruy_MB_dst;

    // Create lhs
    ruy::MakeSimpleLayout(input_shape_MB.size[0], input_shape_MB.size[1], ruy::Order::kColMajor,
                              ruy_MB_lhs.mutable_layout());
    ruy_MB_lhs.set_data(input_data_MB);
    // Create rhs (Column Major)
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kColMajor,
                              ruy_MB_rhs.mutable_layout());
    ruy_MB_rhs.set_data(kernel_data);
    ruy_MB_rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create rhs (Row Major)
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kRowMajor,
                              ruy_MB_rhs_C.mutable_layout());
    ruy_MB_rhs_C.set_data(kernel_data_C);
    ruy_MB_rhs_C.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create dst
    ruy::MakeSimpleLayout(output_shape_MB.size[0], output_shape_MB.size[1], ruy::Order::kColMajor,
                              ruy_MB_dst.mutable_layout());
    ruy_MB_dst.set_data(output_data_R_MB);

    ruy::Context* _ruy_MB_context = new ruy::Context;
    ruy::MulParams<int32_t, int32_t> ruy_MB_mul_params;

    ruy::Mul(ruy_MB_lhs, ruy_MB_rhs, ruy_MB_mul_params, _ruy_MB_context, &ruy_MB_dst);

#if PRINT_VALUES || PRINT_MUL_OUTPUT || PRINT_MUL_MB_OUTPUT
    cout << "[";
    for (int j = 0; j < output_shape_MB.size[0]; j++)
        for (int i = 0; i < output_shape_MB.size[1]; i++)
            cout << ((int)output_data_R_MB[j * output_shape_MB.size[1] + i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    zero_vector(output_data_MB, output_shape_MB.flatsize);
    ret = LowPrecision::FullyConnected::Int4InputsInt8Weights::MultiplyInt8MultiBatched(
                            activation_data_MB, input_shape_MB,
                            filter_data, kernel_shape,
                            output_data_MB, output_shape_MB);
    bool MI8I4MBPassed = true;
    for (int i = 0 ; i < output_shape_MB.flatsize / 2 ; i++)
        MI8I4MBPassed &= (output_data_R_MB[i] == output_data_MB[i]);
    cout << "LowPrecision::FullyConnected::Int4InputsInt8Weights::MultiplyInt8MultiBatched Return Status\t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    cout << "LowPrecision::FullyConnected::Int4InputsInt8Weights::MultiplyInt8MultiBatched\t\t\t=> " << ((MI8I4MBPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (MI8I4MBPassed?"PASSED":"FAILED") << "\033[0m" << endl;
#if PRINT_VALUES || PRINT_MUL_OUTPUT || PRINT_MUL_MB_OUTPUT
    cout << "[";
    for (int j = 0; j < output_shape_MB.size[0]; j++)
        for (int i = 0; i < output_shape_MB.size[1]; i++)
            cout << ((int)output_data_MB[j * output_shape_MB.size[1] + i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    deallocate(input_data);
    deallocate(input_data_MB);
    deallocate(activation_data);
    deallocate(activation_data_MB);
    deallocate(kernel_data);
    deallocate(kernel_data_C);
    deallocate(filter_data);
    deallocate(output_data);
    deallocate(output_data_MB);
    deallocate(output_data_R);
    deallocate(output_data_R_MB);
    cout << "LowPrecision::FullyConnected::Int4InputsInt8Weights  Deallocation\t\t\t\t=> \033[1m\033[32mPASSED\033[0m" << endl;
}

void run_i4i4_tests(
    const int* _template,
    const int8_t* _answers,
    const int kernel_fill_mode,
    const int num_inputs,
    const int num_output,
    const int num_batch
    ){
    int _input_shape[1]         = { num_inputs },
        _activation_shape[1]    = { num_inputs / 2 },
        _input_shape_MB[2]      = { num_batch , num_inputs },
        _activation_MB_shape[2] = { num_batch , num_inputs / 2 },
        _kernel_shape[2]        = { num_output, num_inputs },
        _filter_shape[2]        = { num_output, num_inputs / 2 },
        _output_shape[1]        = { num_output },
        _output_shape_MB[2]     = { num_batch , num_output };
    Shape input_shape           = get_shape(_input_shape,      1),
          activation_shape      = get_shape(_activation_shape, 1),
          input_shape_MB        = get_shape(_input_shape_MB,   2),
          activation_shape_MB   = get_shape(_activation_MB_shape, 2),
          kernel_shape          = get_shape(_kernel_shape,     2),
          filter_shape          = get_shape(_filter_shape,     2),
          output_shape          = get_shape(_output_shape,     1),
          output_shape_MB       = get_shape(_output_shape_MB,  2);
    int8_t*  input_data         = allocate<int8_t>(input_shape.flatsize);
    int8_t*  activation_data    = allocate<int8_t>(activation_shape.flatsize);
    int8_t*  input_data_MB      = allocate<int8_t>(input_shape_MB.flatsize);
    int8_t*  activation_data_MB = allocate<int8_t>(activation_shape_MB.flatsize);
    int8_t*  kernel_data        = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  kernel_data_C      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  filter_data        = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output_data        = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_MB     = allocate<int32_t>(output_shape_MB.flatsize);
    int32_t* output_data_R      = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_R_MB   = allocate<int32_t>(output_shape_MB.flatsize);

    one_vector(input_data, num_inputs);
    // half_one_half_zero_vector(input_data, num_inputs);
    
    for (int i = 0; i < num_batch; i++)
        for (int j = 0; j < num_inputs; j++)
            input_data_MB[i * num_inputs + j] = 1;
    
    if(kernel_fill_mode == 0){
        for (int i = 0; i < kernel_shape.size[0]; i++)
            for (int j = 0; j < kernel_shape.size[1]; j++)
                kernel_data[i * kernel_shape.size[1] + j]   = _template[j % 32];

        for (int i = 0; i < kernel_shape.size[1]; i++)
            for (int j = 0; j < kernel_shape.size[0]; j++)
                kernel_data_C[i * kernel_shape.size[0] + j] = _template[i % 32];
    }
    else if (kernel_fill_mode == 1){
        zero_vector(kernel_data, kernel_shape.flatsize);
        zero_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 2){
        one_vector(kernel_data, kernel_shape.flatsize);
        one_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 3){
        minus_one_vector(kernel_data, kernel_shape.flatsize);
        minus_one_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 4){
        two_vector(kernel_data, kernel_shape.flatsize);
        two_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 5){
        minus_two_vector(kernel_data, kernel_shape.flatsize);
        minus_two_vector(kernel_data_C, kernel_shape.flatsize);
    }

    Status i_ret = LowPrecision::FullyConnected::Int4InputsInt4Weights::QuantizeInput(input_data, input_shape, activation_data, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::Int4InputsInt4Weights::QuantizeInput-RowMajor Return Status\t=> " << ((i_ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((i_ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (activation_shape.flatsize <= 512 && !i_ret){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < activation_shape.flatsize ; i++)
            QFTI4RMPassed &= (activation_data[i] == 17);
        cout << "LowPrecision::FullyConnected::Int4InputsInt4Weights::QuantizeInput-RowMajor\t\t\t=> " << ((QFTI4RMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < 1; i++){
        cout << "\t[";
        for (int j = 0; j < activation_shape.size[0]; j++)
            cout << ((int)activation_data[i * activation_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < 1; i++){
        cout << "\t[";
        for (int j = 0; j < activation_shape.size[0]; j++)
            cout << ((((uint8_t)activation_data[i * activation_shape.size[0] + j]) < 16)?("0"):("")) << (((int)activation_data[i * activation_shape.size[0] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    
    i_ret = LowPrecision::FullyConnected::Int4InputsInt4Weights::QuantizeInput(input_data_MB, input_shape_MB, activation_data_MB, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::Int4InputsInt4Weights::QuantizeInput-MB-RowMajor Return Status\t=> " << ((i_ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((i_ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (activation_shape_MB.flatsize <= 32 * 4 && !i_ret){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < activation_shape_MB.flatsize ; i++){
            if (activation_data_MB[i] != 17)
                cout << i << ": " << activation_data_MB[i] << " -> " << 17 << endl;
            QFTI4RMPassed &= (activation_data_MB[i] == 17);
        }
        cout << "LowPrecision::FullyConnected::Int4InputsInt4Weights::QuantizeInput-MB-RowMajor\t\t\t=> " << ((QFTI4RMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_VALUES || PRINT_ACTIVATIONS
    cout << LowPrecision::get_shape_string(input_shape_MB) << ", " << input_shape_MB.flatsize << endl;
    cout << "[" << endl;
    for (int i = 0; i < input_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape_MB.size[1]; j++)
            cout << ((int)activation_data_MB[i * input_shape_MB.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < activation_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < activation_shape_MB.size[1]; j++)
            cout << ((((uint8_t)activation_data_MB[i * activation_shape_MB.size[0] + j]) < 16)?("0"):("")) << (((int)activation_data_MB[i * activation_shape_MB.size[0] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif

#if PRINT_VALUES
    cout << "Kernel = [" << endl;
    for (int i = 0; i < kernel_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < kernel_shape.size[1]; j++)
            cout << ((int)kernel_data[i * kernel_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "Kernel(0x) = [" << hex << endl;
    for (int i = 0; i < kernel_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < kernel_shape.size[1]; j++)
            cout << (((int)kernel_data[i * kernel_shape.size[1] + j]) & 0x0f) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
#if PRINT_VALUES
    cout << "Kernel_C = [" << endl;
    for (int j = 0; j < kernel_shape.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < kernel_shape.size[1]; i++)
            cout << ((int)kernel_data_C[i * kernel_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "Kernel_C(0x) = [" << hex << endl;
    for (int j = 0; j < kernel_shape.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < kernel_shape.size[1]; i++)
            cout << (((int)kernel_data_C[i * kernel_shape.size[0] + j]) & 0x0f) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    Status ret = LowPrecision::FullyConnected::Int4InputsInt4Weights::QuantizeFilter(kernel_data, kernel_shape, filter_data, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::Int4InputsInt4Weights::QuantizeFilter-RowMajor Return Status \t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 512 && !ret){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4RMPassed &= (kernel_fill_mode == 0 && _answers[i] == filter_data[i]) || (kernel_fill_mode == 2 && filter_data[i] == 17);
        cout << "LowPrecision::FullyConnected::Int4InputsInt4Weights::QuantizeFilter-RowMajor\t\t\t=> " << ((QFTI4RMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((((uint8_t)filter_data[i * filter_shape.size[1] + j]) < 16)?("0"):("")) << (((int)filter_data[i * filter_shape.size[1] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    ret = LowPrecision::FullyConnected::Int4InputsInt4Weights::QuantizeFilter(kernel_data_C, kernel_shape, filter_data, MemLayout::kColumnMajor);
    cout << "LowPrecision::FullyConnected::Int4InputsInt4Weights::QuantizeFilter-ColumnMajor Return Status\t=> " << ((ret)?((ret == LowPrecision::Status::WrongMemLayout)?("\033[1m\033[33m"):("\033[1m\033[31m")):("\033[1m\033[32m")) << ((ret)?((ret == LowPrecision::Status::WrongMemLayout)?("NO_SUPPORT"):("FAILED")):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 512 && !ret){
        bool QFTI4CMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4CMPassed &= (_answers[i] == filter_data[i]);
        cout << "LowPrecision::FullyConnected::Int4InputsInt4Weights::QuantizeFilter-ColumnMajor\t\t\t=> " << ((QFTI4CMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4CMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((((uint8_t)filter_data[i * filter_shape.size[1] + j]) < 16)?("0"):("")) << (((int)filter_data[i * filter_shape.size[1] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    minus_one_vector(output_data_R, output_shape.flatsize);

    ruy::Matrix<int8_t> ruy_lhs;
    ruy::Matrix<int8_t> ruy_rhs;
    ruy::Matrix<int8_t> ruy_rhs_C;
    ruy::Matrix<int32_t> ruy_dst;

    // Create lhs
    ruy::MakeSimpleLayout(1, input_shape.size[0], ruy::Order::kColMajor,
                              ruy_lhs.mutable_layout());
    ruy_lhs.set_data(input_data);
    // Create rhs
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kColMajor,
                              ruy_rhs.mutable_layout());
    ruy_rhs.set_data(kernel_data);
    ruy_rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create rhs
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kRowMajor,
                              ruy_rhs_C.mutable_layout());
    ruy_rhs_C.set_data(kernel_data_C);
    ruy_rhs_C.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create dst
    ruy::MakeSimpleLayout(1, output_shape.size[0], ruy::Order::kColMajor,
                              ruy_dst.mutable_layout());
    ruy_dst.set_data(output_data_R);

    ruy::Context* _ruy_context = new ruy::Context;
    ruy::MulParams<int32_t, int32_t> ruy_mul_params;

    ruy::Mul(ruy_lhs, ruy_rhs, ruy_mul_params, _ruy_context, &ruy_dst);

#if PRINT_VALUES || PRINT_MUL_OUTPUT
    cout << "[";
    for (int i = 0; i < output_shape.size[0]; i++)
        cout << ((int)output_data_R[i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    minus_one_vector(output_data, output_shape.flatsize);
    ret = LowPrecision::FullyConnected::Int4InputsInt4Weights::MultiplyInt8SingleBatch(
                            activation_data, input_shape,
                            filter_data, kernel_shape,
                            output_data, output_shape);
    bool MI8I4Passed = true;
    for (int i = 0 ; i < output_shape.flatsize / 2 ; i++)
        MI8I4Passed &= (output_data_R[i] == output_data[i]);

    cout << "LowPrecision::FullyConnected::Int4InputsInt4Weights::MultiplyInt8SingleBatch Return Status\t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    cout << "LowPrecision::FullyConnected::Int4InputsInt4Weights::MultiplyInt8SingleBatch\t\t\t=> " << ((MI8I4Passed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (MI8I4Passed?"PASSED":"FAILED") << "\033[0m" << endl;
#if PRINT_VALUES || PRINT_MUL_OUTPUT
    cout << "[";
    for (int i = 0; i < output_shape.size[0]; i++)
        cout << ((int)output_data[i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    zero_vector(output_data_R_MB, output_shape_MB.flatsize);

    ruy::Matrix<int8_t> ruy_MB_lhs;
    ruy::Matrix<int8_t> ruy_MB_rhs;
    ruy::Matrix<int8_t> ruy_MB_rhs_C;
    ruy::Matrix<int32_t> ruy_MB_dst;

    // Create lhs
    ruy::MakeSimpleLayout(input_shape_MB.size[0], input_shape_MB.size[1], ruy::Order::kColMajor,
                              ruy_MB_lhs.mutable_layout());
    ruy_MB_lhs.set_data(input_data_MB);
    // Create rhs (Column Major)
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kColMajor,
                              ruy_MB_rhs.mutable_layout());
    ruy_MB_rhs.set_data(kernel_data);
    ruy_MB_rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create rhs (Row Major)
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kRowMajor,
                              ruy_MB_rhs_C.mutable_layout());
    ruy_MB_rhs_C.set_data(kernel_data_C);
    ruy_MB_rhs_C.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create dst
    ruy::MakeSimpleLayout(output_shape_MB.size[0], output_shape_MB.size[1], ruy::Order::kColMajor,
                              ruy_MB_dst.mutable_layout());
    ruy_MB_dst.set_data(output_data_R_MB);

    ruy::Context* _ruy_MB_context = new ruy::Context;
    ruy::MulParams<int32_t, int32_t> ruy_MB_mul_params;

    ruy::Mul(ruy_MB_lhs, ruy_MB_rhs, ruy_MB_mul_params, _ruy_MB_context, &ruy_MB_dst);

#if PRINT_VALUES || PRINT_MUL_OUTPUT || PRINT_MUL_MB_OUTPUT
    cout << "[";
    for (int j = 0; j < output_shape_MB.size[0]; j++)
        for (int i = 0; i < output_shape_MB.size[1]; i++)
            cout << ((int)output_data_R_MB[j * output_shape_MB.size[1] + i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    zero_vector(output_data_MB, output_shape_MB.flatsize);
    ret = LowPrecision::FullyConnected::Int4InputsInt4Weights::MultiplyInt8MultiBatched(
                            activation_data_MB, input_shape_MB,
                            filter_data, kernel_shape,
                            output_data_MB, output_shape_MB);
    bool MI8I4MBPassed = true;
    for (int i = 0 ; i < output_shape_MB.flatsize / 2 ; i++)
        MI8I4MBPassed &= (output_data_R_MB[i] == output_data_MB[i]);
    cout << "LowPrecision::FullyConnected::Int4InputsInt4Weights::MultiplyInt8MultiBatched Return Status\t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    cout << "LowPrecision::FullyConnected::Int4InputsInt4Weights::MultiplyInt8MultiBatched\t\t\t=> " << ((MI8I4MBPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (MI8I4MBPassed?"PASSED":"FAILED") << "\033[0m" << endl;
#if PRINT_VALUES || PRINT_MUL_OUTPUT || PRINT_MUL_MB_OUTPUT
    cout << "[";
    for (int j = 0; j < output_shape_MB.size[0]; j++)
        for (int i = 0; i < output_shape_MB.size[1]; i++)
            cout << ((int)output_data_MB[j * output_shape_MB.size[1] + i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    deallocate(input_data);
    deallocate(input_data_MB);
    deallocate(activation_data);
    deallocate(activation_data_MB);
    deallocate(kernel_data);
    deallocate(kernel_data_C);
    deallocate(filter_data);
    deallocate(output_data);
    deallocate(output_data_MB);
    deallocate(output_data_R);
    deallocate(output_data_R_MB);
    cout << "LowPrecision::FullyConnected::Int4InputsInt4Weights  Deallocation\t\t\t\t=> \033[1m\033[32mPASSED\033[0m" << endl;
}

void run_teri8_tests(
    const int* _template,
    const int8_t* _answers,
    const int kernel_fill_mode,
    const int num_inputs,
    const int num_output,
    const int num_batch
    ){
    int _input_shape[1]         = { num_inputs },
        _activation_shape[1]    = { num_inputs / 4 },
        _input_shape_MB[2]      = { num_batch , num_inputs },
        _activation_MB_shape[2] = { num_batch , num_inputs / 4 },
        _kernel_shape[2]        = { num_output, num_inputs },
        _filter_shape[2]        = { num_output, num_inputs },
        _output_shape[1]        = { num_output },
        _output_shape_MB[2]     = { num_batch , num_output };
    Shape input_shape           = get_shape(_input_shape,         1),
          activation_shape      = get_shape(_activation_shape,    1),
          input_shape_MB        = get_shape(_input_shape_MB,      2),
          activation_shape_MB   = get_shape(_activation_MB_shape, 2),
          kernel_shape          = get_shape(_kernel_shape,        2),
          filter_shape          = get_shape(_filter_shape,        2),
          output_shape          = get_shape(_output_shape,        1),
          output_shape_MB       = get_shape(_output_shape_MB,     2);
    int8_t*  input_data         = allocate<int8_t>(input_shape.flatsize);
    int8_t*  activation_data    = allocate<int8_t>(activation_shape.flatsize);
    int8_t*  input_data_MB      = allocate<int8_t>(input_shape_MB.flatsize);
    int8_t*  activation_data_MB = allocate<int8_t>(activation_shape_MB.flatsize);
    int8_t*  kernel_data        = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  kernel_data_C      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  filter_data        = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output_data        = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_MB     = allocate<int32_t>(output_shape_MB.flatsize);
    int32_t* output_data_R      = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_R_MB   = allocate<int32_t>(output_shape_MB.flatsize);

    // one_vector(input_data, num_inputs);
    // half_one_half_zero_vector(input_data, num_inputs);
    int     input_template[]        = { -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0 };
    uint8_t activation_answers_u[]  = { 0xff, 0x55, 0x00, 0xff, 0x55, 0x00, 0xff, 0x55, 0x00, 0xff, 0x55, 0x00, 0xff, 0x55, 0x00, 0x00 };
    int8_t* activation_answers      = LowPrecision::get_pointer_as<int8_t>(activation_answers_u);
    for (int i = 0; i < num_inputs; i++)
        input_data[i] = input_template[i % 16];
    
    for (int i = 0; i < num_batch; i++)
        for (int j = 0; j < num_inputs; j++)
            input_data_MB[i * num_inputs + j] = input_template[j % 16];
    
    if(kernel_fill_mode == 0){
        for (int i = 0; i < kernel_shape.size[0]; i++)
            for (int j = 0; j < kernel_shape.size[1]; j++)
                kernel_data[i * kernel_shape.size[1] + j]   = _template[j % 32];

        for (int i = 0; i < kernel_shape.size[1]; i++)
            for (int j = 0; j < kernel_shape.size[0]; j++)
                kernel_data_C[i * kernel_shape.size[0] + j] = _template[i % 32];
    }
    else if (kernel_fill_mode == 1){
        zero_vector(kernel_data, kernel_shape.flatsize);
        zero_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 2){
        one_vector(kernel_data, kernel_shape.flatsize);
        one_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 3){
        minus_one_vector(kernel_data, kernel_shape.flatsize);
        minus_one_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 4){
        two_vector(kernel_data, kernel_shape.flatsize);
        two_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 5){
        minus_two_vector(kernel_data, kernel_shape.flatsize);
        minus_two_vector(kernel_data_C, kernel_shape.flatsize);
    }

#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < 1; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape.size[0]; j++)
            cout << ((int)input_data[i * input_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif

    Status i_ret = LowPrecision::FullyConnected::TernaryInputsInt8Weights::QuantizeInput(input_data, input_shape, activation_data, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::TernaryInputsInt8Weights::QuantizeInput-RowMajor Return Status\t=> " << ((i_ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((i_ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (activation_shape.flatsize <= 512 && i_ret){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < activation_shape.flatsize ; i++)
            QFTI4RMPassed &= (activation_data[i] == activation_answers[i % 16]);
        cout << "LowPrecision::FullyConnected::TernaryInputsInt8Weights::QuantizeInput-RowMajor\t\t\t=> " << ((QFTI4RMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < 1; i++){
        cout << "\t[";
        for (int j = 0; j < activation_shape.size[0]; j++)
            cout << ((int)activation_data[i * activation_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < 1; i++){
        cout << "\t[";
        for (int j = 0; j < activation_shape.size[0]; j++)
            cout << ((((uint8_t)activation_data[i * activation_shape.size[0] + j]) < 16)?("0"):("")) << (((int)activation_data[i * activation_shape.size[0] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif

#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < input_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape_MB.size[1]; j++)
            cout << ((int)input_data_MB[i * input_shape_MB.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif

    i_ret = LowPrecision::FullyConnected::TernaryInputsInt8Weights::QuantizeInput(input_data_MB, input_shape_MB, activation_data_MB, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::TernaryInputsInt8Weights::QuantizeInput-MB-RowMajor Return Status\t=> " << ((i_ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((i_ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (activation_shape_MB.flatsize <= 32 * 4 && !i_ret){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < activation_shape_MB.flatsize ; i++){
            if (activation_data_MB[i] != activation_answers[i % 16])
                cout << i << ": " << ((int)activation_data_MB[i]) << " -> " << ((int)activation_answers[i % 16]) << endl;
            QFTI4RMPassed &= (activation_data_MB[i] == activation_answers[i % 16]);
        }
        cout << "LowPrecision::FullyConnected::TernaryInputsInt8Weights::QuantizeInput-MB-RowMajor\t\t=> " << ((QFTI4RMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_VALUES || PRINT_ACTIVATIONS
    cout << LowPrecision::get_shape_string(input_shape_MB) << ", " << input_shape_MB.flatsize << endl;
    cout << "[" << endl;
    for (int i = 0; i < input_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape_MB.size[1]; j++)
            cout << ((int)activation_data_MB[i * input_shape_MB.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < activation_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < activation_shape_MB.size[1]; j++)
            cout << ((((uint8_t)activation_data_MB[i * activation_shape_MB.size[0] + j]) < 16)?("0"):("")) << (((int)activation_data_MB[i * activation_shape_MB.size[0] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif

#if PRINT_VALUES
    cout << "Kernel = [" << endl;
    for (int i = 0; i < kernel_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < kernel_shape.size[1]; j++)
            cout << ((int)kernel_data[i * kernel_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "Kernel(0x) = [" << hex << endl;
    for (int i = 0; i < kernel_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < kernel_shape.size[1]; j++)
            cout << (((int)kernel_data[i * kernel_shape.size[1] + j]) & 0x0f) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
#if PRINT_VALUES
    cout << "Kernel_C = [" << endl;
    for (int j = 0; j < kernel_shape.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < kernel_shape.size[1]; i++)
            cout << ((int)kernel_data_C[i * kernel_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "Kernel_C(0x) = [" << hex << endl;
    for (int j = 0; j < kernel_shape.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < kernel_shape.size[1]; i++)
            cout << (((int)kernel_data_C[i * kernel_shape.size[0] + j]) & 0x0f) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    Status ret = LowPrecision::FullyConnected::TernaryInputsInt8Weights::QuantizeFilter(kernel_data, kernel_shape, filter_data, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::TernaryInputsInt8Weights::QuantizeFilter-RowMajor Return Status\t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 4096 && !ret){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4RMPassed &= (filter_data[i] == 1);
        cout << "LowPrecision::FullyConnected::TernaryInputsInt8Weights::QuantizeFilter-RowMajor\t\t\t=> " << ((QFTI4RMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((((uint8_t)filter_data[i * filter_shape.size[1] + j]) < 16)?("0"):("")) << (((int)filter_data[i * filter_shape.size[1] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    ret = LowPrecision::FullyConnected::TernaryInputsInt8Weights::QuantizeFilter(kernel_data_C, kernel_shape, filter_data, MemLayout::kColumnMajor);
    cout << "LowPrecision::FullyConnected::TernaryInputsInt8Weights::QuantizeFilter-ColumnMajor Return Statu\t=> " << ((ret)?((ret == LowPrecision::Status::WrongMemLayout)?("\033[1m\033[33m"):("\033[1m\033[31m")):("\033[1m\033[32m")) << ((ret)?((ret == LowPrecision::Status::WrongMemLayout)?("NO_SUPPORT"):("FAILED")):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 4096 && !ret){
        bool QFTI4CMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4CMPassed &= (filter_data[i] == 1);
        cout << "LowPrecision::FullyConnected::TernaryInputsInt8Weights::QuantizeFilter-ColumnMajor\t\t\t=> " << ((QFTI4CMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4CMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((((uint8_t)filter_data[i * filter_shape.size[1] + j]) < 16)?("0"):("")) << (((int)filter_data[i * filter_shape.size[1] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    minus_one_vector(output_data_R, output_shape.flatsize);

    ruy::Matrix<int8_t> ruy_lhs;
    ruy::Matrix<int8_t> ruy_rhs;
    ruy::Matrix<int8_t> ruy_rhs_C;
    ruy::Matrix<int32_t> ruy_dst;

    // Create lhs
    ruy::MakeSimpleLayout(1, input_shape.size[0], ruy::Order::kColMajor,
                              ruy_lhs.mutable_layout());
    ruy_lhs.set_data(input_data);
    // Create rhs
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kColMajor,
                              ruy_rhs.mutable_layout());
    ruy_rhs.set_data(kernel_data);
    ruy_rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create rhs
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kRowMajor,
                              ruy_rhs_C.mutable_layout());
    ruy_rhs_C.set_data(kernel_data_C);
    ruy_rhs_C.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create dst
    ruy::MakeSimpleLayout(1, output_shape.size[0], ruy::Order::kColMajor,
                              ruy_dst.mutable_layout());
    ruy_dst.set_data(output_data_R);

    ruy::Context* _ruy_context = new ruy::Context;
    ruy::MulParams<int32_t, int32_t> ruy_mul_params;

    ruy::Mul(ruy_lhs, ruy_rhs, ruy_mul_params, _ruy_context, &ruy_dst);

#if PRINT_VALUES || PRINT_MUL_OUTPUT
    cout << "[";
    for (int i = 0; i < output_shape.size[0]; i++)
        cout << ((int)output_data_R[i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    minus_one_vector(output_data, output_shape.flatsize);
    ret = LowPrecision::FullyConnected::TernaryInputsInt8Weights::MultiplyInt8SingleBatch(
                            activation_data, input_shape,
                            filter_data, kernel_shape,
                            output_data, output_shape);
    bool MI8I4Passed = true;
    for (int i = 0 ; i < output_shape.flatsize / 2 ; i++)
        MI8I4Passed &= (output_data_R[i] == output_data[i]);

    cout << "LowPrecision::FullyConnected::TernaryInputsInt8Weights::MultiplyInt8SingleBatch Return Status\t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    cout << "LowPrecision::FullyConnected::TernaryInputsInt8Weights::MultiplyInt8SingleBatch\t\t\t=> " << ((MI8I4Passed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (MI8I4Passed?"PASSED":"FAILED") << "\033[0m" << endl;
#if PRINT_VALUES || PRINT_MUL_OUTPUT
    cout << "[";
    for (int i = 0; i < output_shape.size[0]; i++)
        cout << ((int)output_data[i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    zero_vector(output_data_R_MB, output_shape_MB.flatsize);

    ruy::Matrix<int8_t>  ruy_MB_lhs;
    ruy::Matrix<int8_t>  ruy_MB_rhs;
    ruy::Matrix<int32_t> ruy_MB_dst;

    // Create lhs
    ruy::MakeSimpleLayout(input_shape_MB.size[0], input_shape_MB.size[1], ruy::Order::kRowMajor,
                              ruy_MB_lhs.mutable_layout());
    ruy_MB_lhs.set_data(input_data_MB);
    // Create rhs (Column Major)
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kColMajor,
                              ruy_MB_rhs.mutable_layout());
    ruy_MB_rhs.set_data(kernel_data);
    ruy_MB_rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create dst
    ruy::MakeSimpleLayout(output_shape_MB.size[0], output_shape_MB.size[1], ruy::Order::kColMajor,
                              ruy_MB_dst.mutable_layout());
    ruy_MB_dst.set_data(output_data_R_MB);

    ruy::Context* _ruy_MB_context = new ruy::Context;
    ruy::MulParams<int32_t, int32_t> ruy_MB_mul_params;

    ruy::Mul(ruy_MB_lhs, ruy_MB_rhs, ruy_MB_mul_params, _ruy_MB_context, &ruy_MB_dst);

#if PRINT_VALUES || PRINT_MUL_OUTPUT || PRINT_MUL_MB_OUTPUT
    cout << "[";
    for (int j = 0; j < output_shape_MB.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < output_shape_MB.size[1]; i++)
            cout << ((int)output_data_R_MB[j * output_shape_MB.size[1] + i]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif

    zero_vector(output_data_MB, output_shape_MB.flatsize);
    ret = LowPrecision::FullyConnected::TernaryInputsInt8Weights::MultiplyInt8MultiBatched(
                            activation_data_MB, input_shape_MB,
                            filter_data, kernel_shape,
                            output_data_MB, output_shape_MB);
    bool MI8I4MBPassed = true;
    for (int i = 0 ; i < output_shape_MB.flatsize / 2 ; i++)
        MI8I4MBPassed &= (output_data_R_MB[i] == output_data_MB[i]);
    cout << "LowPrecision::FullyConnected::TernaryInputsInt8Weights::MultiplyInt8MultiBatched Return Status\t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    cout << "LowPrecision::FullyConnected::TernaryInputsInt8Weights::MultiplyInt8MultiBatched\t\t=> " << ((MI8I4MBPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (MI8I4MBPassed?"PASSED":"FAILED") << "\033[0m" << endl;
#if PRINT_VALUES || PRINT_MUL_OUTPUT || PRINT_MUL_MB_OUTPUT
    cout << "[";
    for (int j = 0; j < output_shape_MB.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < output_shape_MB.size[1]; i++)
            cout << ((int)output_data_MB[j * output_shape_MB.size[1] + i]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif

    deallocate(input_data);
    deallocate(input_data_MB);
    deallocate(activation_data_MB);
    deallocate(kernel_data);
    deallocate(kernel_data_C);
    deallocate(filter_data);
    deallocate(output_data);
    deallocate(output_data_MB);
    deallocate(output_data_R);
    deallocate(output_data_R_MB);
    cout << "LowPrecision::FullyConnected::TernaryInputsInt8Weights  Deallocation\t\t\t\t=> \033[1m\033[32mPASSED\033[0m" << endl;
}

void run_bini8_tests(
    const int* _template,
    const int8_t* _answers,
    const int kernel_fill_mode,
    const int num_inputs,
    const int num_output,
    const int num_batch
    ){
    int _input_shape[1]       = { num_inputs },
        _activation_shape[1]  = { num_inputs / 8 },
        _input_shape_MB[2]    = { num_batch , num_inputs },
        _kernel_shape[2]      = { num_output, num_inputs },
        _filter_shape[2]      = { num_output, num_inputs },
        _output_shape[1]      = { num_output },
        _output_shape_MB[2]   = { num_batch , num_output };
    Shape input_shape         = get_shape(_input_shape,      1),
          activation_shape    = get_shape(_activation_shape, 1),
          input_shape_MB      = get_shape(_input_shape_MB,   2),
          kernel_shape        = get_shape(_kernel_shape,     2),
          filter_shape        = get_shape(_filter_shape,     2),
          output_shape        = get_shape(_output_shape,     1),
          output_shape_MB     = get_shape(_output_shape_MB,  2);
    
    int8_t*  input_data       = allocate<int8_t>(input_shape.flatsize);
    int8_t*  activation_data  = allocate<int8_t>(activation_shape.flatsize);
    int8_t*  input_data_MB    = allocate<int8_t>(input_shape_MB.flatsize);
    int8_t*  input_pack_MB    = allocate<int8_t>(input_shape_MB.flatsize);
    int8_t*  kernel_data      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  kernel_data_C    = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  filter_data      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output_data      = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_MB   = allocate<int32_t>(output_shape_MB.flatsize);
    int32_t* output_data_R    = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_R_MB = allocate<int32_t>(output_shape_MB.flatsize);

    // one_vector(input_data, num_inputs);
    // half_one_half_zero_vector(input_data, num_inputs);
    int     input_template[]        = { -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1 };
    uint8_t activation_answers_u[]  = { 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, };
    int8_t* activation_answers      = LowPrecision::get_pointer_as<int8_t>(activation_answers_u);
    for (int i = 0; i < num_inputs; i++)
        input_data[i] = input_template[i % 16];
    
    for (int i = 0; i < num_batch; i++)
        for (int j = 0; j < num_inputs; j++)
            input_data_MB[i * num_inputs + j] = 1;
    
    if(kernel_fill_mode == 0){
        for (int i = 0; i < kernel_shape.size[0]; i++)
            for (int j = 0; j < kernel_shape.size[1]; j++)
                kernel_data[i * kernel_shape.size[1] + j]   = _template[j % 32];

        for (int i = 0; i < kernel_shape.size[1]; i++)
            for (int j = 0; j < kernel_shape.size[0]; j++)
                kernel_data_C[i * kernel_shape.size[0] + j] = _template[i % 32];
    }
    else if (kernel_fill_mode == 1){
        zero_vector(kernel_data, kernel_shape.flatsize);
        zero_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 2){
        one_vector(kernel_data, kernel_shape.flatsize);
        one_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 3){
        minus_one_vector(kernel_data, kernel_shape.flatsize);
        minus_one_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 4){
        two_vector(kernel_data, kernel_shape.flatsize);
        two_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 5){
        minus_two_vector(kernel_data, kernel_shape.flatsize);
        minus_two_vector(kernel_data_C, kernel_shape.flatsize);
    }

#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < 1; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape.size[0]; j++)
            cout << ((int)input_data[i * input_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif

    Status i_ret = LowPrecision::FullyConnected::BinaryInputsInt8Weights::QuantizeInput(input_data, input_shape, activation_data, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::BinaryInputsInt8Weights::QuantizeInput-RowMajor Return Status\t=> " << ((i_ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((i_ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (activation_shape.flatsize <= 512 && i_ret){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < activation_shape.flatsize ; i++)
            QFTI4RMPassed &= (activation_data[i] == activation_answers[i % 16]);
        cout << "LowPrecision::FullyConnected::BinaryInputsInt8Weights::QuantizeInput-RowMajor\t\t\t=> " << ((QFTI4RMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < 1; i++){
        cout << "\t[";
        for (int j = 0; j < activation_shape.size[0]; j++)
            cout << ((int)activation_data[i * activation_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < 1; i++){
        cout << "\t[";
        for (int j = 0; j < activation_shape.size[0]; j++)
            cout << ((((uint8_t)activation_data[i * activation_shape.size[0] + j]) < 16)?("0"):("")) << (((int)activation_data[i * activation_shape.size[0] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif

#if PRINT_VALUES
    cout << "Kernel = [" << endl;
    for (int i = 0; i < kernel_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < kernel_shape.size[1]; j++)
            cout << ((int)kernel_data[i * kernel_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "Kernel(0x) = [" << hex << endl;
    for (int i = 0; i < kernel_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < kernel_shape.size[1]; j++)
            cout << (((int)kernel_data[i * kernel_shape.size[1] + j]) & 0x0f) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
#if PRINT_VALUES
    cout << "Kernel_C = [" << endl;
    for (int j = 0; j < kernel_shape.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < kernel_shape.size[1]; i++)
            cout << ((int)kernel_data_C[i * kernel_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "Kernel_C(0x) = [" << hex << endl;
    for (int j = 0; j < kernel_shape.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < kernel_shape.size[1]; i++)
            cout << (((int)kernel_data_C[i * kernel_shape.size[0] + j]) & 0x0f) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    Status ret = LowPrecision::FullyConnected::BinaryInputsInt8Weights::QuantizeFilter(kernel_data, kernel_shape, filter_data, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::BinaryInputsInt8Weights::QuantizeFilter-RowMajor Return Status\t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 4096 && !ret){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4RMPassed &= (filter_data[i] == 1);
        cout << "LowPrecision::FullyConnected::BinaryInputsInt8Weights::QuantizeFilter-RowMajor\t\t\t=> " << ((QFTI4RMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((((uint8_t)filter_data[i * filter_shape.size[1] + j]) < 16)?("0"):("")) << (((int)filter_data[i * filter_shape.size[1] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    ret = LowPrecision::FullyConnected::BinaryInputsInt8Weights::QuantizeFilter(kernel_data_C, kernel_shape, filter_data, MemLayout::kColumnMajor);
    cout << "LowPrecision::FullyConnected::BinaryInputsInt8Weights::QuantizeFilter-ColumnMajor Return Statu\t=> " << ((ret)?((ret == LowPrecision::Status::WrongMemLayout)?("\033[1m\033[33m"):("\033[1m\033[31m")):("\033[1m\033[32m")) << ((ret)?((ret == LowPrecision::Status::WrongMemLayout)?("NO_SUPPORT"):("FAILED")):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 4096 && !ret){
        bool QFTI4CMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4CMPassed &= (filter_data[i] == 1);
        cout << "LowPrecision::FullyConnected::BinaryInputsInt8Weights::QuantizeFilter-ColumnMajor\t\t\t=> " << ((QFTI4CMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4CMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((((uint8_t)filter_data[i * filter_shape.size[1] + j]) < 16)?("0"):("")) << (((int)filter_data[i * filter_shape.size[1] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    minus_one_vector(output_data_R, output_shape.flatsize);

    ruy::Matrix<int8_t> ruy_lhs;
    ruy::Matrix<int8_t> ruy_rhs;
    ruy::Matrix<int8_t> ruy_rhs_C;
    ruy::Matrix<int32_t> ruy_dst;

    // Create lhs
    ruy::MakeSimpleLayout(1, input_shape.size[0], ruy::Order::kColMajor,
                              ruy_lhs.mutable_layout());
    ruy_lhs.set_data(input_data);
    // Create rhs
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kColMajor,
                              ruy_rhs.mutable_layout());
    ruy_rhs.set_data(kernel_data);
    ruy_rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create rhs
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kRowMajor,
                              ruy_rhs_C.mutable_layout());
    ruy_rhs_C.set_data(kernel_data_C);
    ruy_rhs_C.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create dst
    ruy::MakeSimpleLayout(1, output_shape.size[0], ruy::Order::kColMajor,
                              ruy_dst.mutable_layout());
    ruy_dst.set_data(output_data_R);

    ruy::Context* _ruy_context = new ruy::Context;
    ruy::MulParams<int32_t, int32_t> ruy_mul_params;

    ruy::Mul(ruy_lhs, ruy_rhs, ruy_mul_params, _ruy_context, &ruy_dst);

#if PRINT_VALUES || PRINT_MUL_OUTPUT
    cout << "[";
    for (int i = 0; i < output_shape.size[0]; i++)
        cout << ((int)output_data_R[i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    minus_one_vector(output_data, output_shape.flatsize);
    ret = LowPrecision::FullyConnected::BinaryInputsInt8Weights::MultiplyInt8SingleBatch(
                            activation_data, input_shape,
                            filter_data, kernel_shape,
                            output_data, output_shape);
    bool MI8I4Passed = true;
    for (int i = 0 ; i < output_shape.flatsize / 2 ; i++)
        MI8I4Passed &= (output_data_R[i] == output_data[i]);

    cout << "LowPrecision::FullyConnected::BinaryInputsInt8Weights::MultiplyInt8SingleBatch Return Status\t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    cout << "LowPrecision::FullyConnected::BinaryInputsInt8Weights::MultiplyInt8SingleBatch\t\t\t=> " << ((MI8I4Passed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (MI8I4Passed?"PASSED":"FAILED") << "\033[0m" << endl;
#if PRINT_VALUES || PRINT_MUL_OUTPUT
    cout << "[";
    for (int i = 0; i < output_shape.size[0]; i++)
        cout << ((int)output_data[i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    deallocate(input_data);
    deallocate(input_data_MB);
    deallocate(input_pack_MB);
    deallocate(kernel_data);
    deallocate(kernel_data_C);
    deallocate(filter_data);
    deallocate(output_data);
    deallocate(output_data_MB);
    deallocate(output_data_R);
    deallocate(output_data_R_MB);
    cout << "LowPrecision::FullyConnected::BinaryInputsInt8Weights  Deallocation\t\t\t\t=> \033[1m\033[32mPASSED\033[0m" << endl;
}

void run_binbin_tests(
    const int* _template,
    const int8_t* _answers,
    const int kernel_fill_mode,
    const int num_inputs,
    const int num_output,
    const int num_batch
    ){
    int _input_shape[1]       = { num_inputs },
        _activation_shape[1]  = { num_inputs / 8 },
        _input_shape_MB[2]    = { num_batch , num_inputs },
        _kernel_shape[2]      = { num_output, num_inputs },
        _filter_shape[2]      = { num_output, num_inputs / 8 },
        _output_shape[1]      = { num_output },
        _output_shape_MB[2]   = { num_batch , num_output };
    Shape input_shape         = get_shape(_input_shape,      1),
          activation_shape    = get_shape(_activation_shape, 1),
          input_shape_MB      = get_shape(_input_shape_MB,   2),
          kernel_shape        = get_shape(_kernel_shape,     2),
          filter_shape        = get_shape(_filter_shape,     2),
          output_shape        = get_shape(_output_shape,     1),
          output_shape_MB     = get_shape(_output_shape_MB,  2);
    
    int8_t*  input_data       = allocate<int8_t>(input_shape.flatsize);
    int8_t*  activation_data  = allocate<int8_t>(activation_shape.flatsize);
    int8_t*  input_data_MB    = allocate<int8_t>(input_shape_MB.flatsize);
    int8_t*  input_pack_MB    = allocate<int8_t>(input_shape_MB.flatsize);
    int8_t*  kernel_data      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  kernel_data_C    = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  filter_data      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output_data      = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_MB   = allocate<int32_t>(output_shape_MB.flatsize);
    int32_t* output_data_R    = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_R_MB = allocate<int32_t>(output_shape_MB.flatsize);

    int     input_template[]        = { -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1 };
    uint8_t activation_answers_u[]  = { 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, };
    int8_t* activation_answers      = LowPrecision::get_pointer_as<int8_t>(activation_answers_u);
    for (int i = 0; i < num_inputs; i++)
        input_data[i] = input_template[i % 16];
    // one_vector(input_data, num_inputs);
    // half_one_half_zero_vector(input_data, num_inputs);
    
    for (int i = 0; i < num_batch; i++)
        for (int j = 0; j < num_inputs; j++)
            input_data_MB[i * num_inputs + j] = 1;
    
    if(kernel_fill_mode == 0){
        for (int i = 0; i < kernel_shape.size[0]; i++)
            for (int j = 0; j < kernel_shape.size[1]; j++)
                kernel_data[i * kernel_shape.size[1] + j]   = _template[j % 32];

        for (int i = 0; i < kernel_shape.size[1]; i++)
            for (int j = 0; j < kernel_shape.size[0]; j++)
                kernel_data_C[i * kernel_shape.size[0] + j] = _template[i % 32];
    }
    else if (kernel_fill_mode == 1){
        zero_vector(kernel_data, kernel_shape.flatsize);
        zero_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 2){
        one_vector(kernel_data, kernel_shape.flatsize);
        one_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 3){
        minus_one_vector(kernel_data, kernel_shape.flatsize);
        minus_one_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 4){
        two_vector(kernel_data, kernel_shape.flatsize);
        two_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 5){
        minus_two_vector(kernel_data, kernel_shape.flatsize);
        minus_two_vector(kernel_data_C, kernel_shape.flatsize);
    }

#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < 1; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape.size[0]; j++)
            cout << ((int)input_data[i * input_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif

    Status i_ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeights::QuantizeInput(input_data, input_shape, activation_data, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::BinaryInputsBinaryWeights::QuantizeInput-RowMajor Return Status\t=> " << ((i_ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((i_ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (activation_shape.flatsize <= 128 * 128 + 128 && !i_ret){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < activation_shape.flatsize ; i++)
            QFTI4RMPassed &= (activation_data[i] == activation_answers[i % 16]);
        cout << "LowPrecision::FullyConnected::BinaryInputsBinaryWeights::QuantizeInput-RowMajor\t\t\t=> " << ((QFTI4RMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < 1; i++){
        cout << "\t[";
        for (int j = 0; j < activation_shape.size[0]; j++)
            cout << ((int)activation_data[i * activation_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < 1; i++){
        cout << "\t[";
        for (int j = 0; j < activation_shape.size[0]; j++)
            cout << ((((uint8_t)activation_data[i * activation_shape.size[0] + j]) < 16)?("0"):("")) << (((int)activation_data[i * activation_shape.size[0] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif

#if PRINT_KERNEL
    cout << "Kernel = [" << endl;
    for (int i = 0; i < kernel_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < kernel_shape.size[1]; j++)
            cout << ((int)kernel_data[i * kernel_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "Kernel(0x) = [" << hex << endl;
    for (int i = 0; i < kernel_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < kernel_shape.size[1]; j++)
            cout << (((int)kernel_data[i * kernel_shape.size[1] + j]) & 0x0f) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
#if PRINT_KERNEL
    cout << "Kernel_C = [" << endl;
    for (int j = 0; j < kernel_shape.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < kernel_shape.size[1]; i++)
            cout << ((int)kernel_data_C[i * kernel_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "Kernel_C(0x) = [" << hex << endl;
    for (int j = 0; j < kernel_shape.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < kernel_shape.size[1]; i++)
            cout << (((int)kernel_data_C[i * kernel_shape.size[0] + j]) & 0x0f) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    Status ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeights::QuantizeFilter(kernel_data, kernel_shape, filter_data, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::BinaryInputsBinaryWeights::QuantizeFilter-RowMajor Return Status\t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 4096 && !ret){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4RMPassed &= (filter_data[i] == _answers[i]);
        cout << "LowPrecision::FullyConnected::BinaryInputsBinaryWeights::QuantizeFilter-RowMajor\t\t=> " << ((QFTI4RMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((((uint8_t)filter_data[i * filter_shape.size[1] + j]) < 16)?("0"):("")) << (((int)filter_data[i * filter_shape.size[1] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    
    ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeights::QuantizeFilter(kernel_data_C, kernel_shape, filter_data, MemLayout::kColumnMajor);
    cout << "LowPrecision::FullyConnected::BinaryInputsBinaryWeights::QuantizeFilter-ColumnMajor Return Statu=> " << ((ret)?((ret == LowPrecision::Status::WrongMemLayout)?("\033[1m\033[33m"):("\033[1m\033[31m")):("\033[1m\033[32m")) << ((ret)?((ret == LowPrecision::Status::WrongMemLayout)?("NO_SUPPORT"):("FAILED")):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 4096 && !ret){
        bool QFTI4CMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4CMPassed &= (filter_data[i] == _answers[i]);
        cout << "LowPrecision::FullyConnected::BinaryInputsBinaryWeights::QuantizeFilter-ColumnMajor\t\t\t=> " << ((QFTI4CMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4CMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((((uint8_t)filter_data[i * filter_shape.size[1] + j]) < 16)?("0"):("")) << (((int)filter_data[i * filter_shape.size[1] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    minus_one_vector(output_data_R, output_shape.flatsize);

    ruy::Matrix<int8_t> ruy_lhs;
    ruy::Matrix<int8_t> ruy_rhs;
    ruy::Matrix<int8_t> ruy_rhs_C;
    ruy::Matrix<int32_t> ruy_dst;

    // Create lhs
    ruy::MakeSimpleLayout(1, input_shape.size[0], ruy::Order::kColMajor,
                              ruy_lhs.mutable_layout());
    ruy_lhs.set_data(input_data);
    // Create rhs
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kColMajor,
                              ruy_rhs.mutable_layout());
    ruy_rhs.set_data(kernel_data);
    ruy_rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create rhs
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kRowMajor,
                              ruy_rhs_C.mutable_layout());
    ruy_rhs_C.set_data(kernel_data_C);
    ruy_rhs_C.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create dst
    ruy::MakeSimpleLayout(1, output_shape.size[0], ruy::Order::kColMajor,
                              ruy_dst.mutable_layout());
    ruy_dst.set_data(output_data_R);

    ruy::Context* _ruy_context = new ruy::Context;
    ruy::MulParams<int32_t, int32_t> ruy_mul_params;

    ruy::Mul(ruy_lhs, ruy_rhs, ruy_mul_params, _ruy_context, &ruy_dst);

#if PRINT_VALUES || PRINT_MUL_OUTPUT
    cout << "[";
    for (int i = 0; i < output_shape.size[0]; i++)
        cout << ((int)output_data_R[i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    minus_one_vector(output_data, output_shape.flatsize);
    ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeights::MultiplyInt8SingleBatch(
                            activation_data, input_shape,
                            filter_data, kernel_shape,
                            output_data, output_shape);
    bool MI8I4Passed = true;
    for (int i = 0 ; i < output_shape.flatsize / 2 ; i++)
        MI8I4Passed &= (output_data_R[i] == output_data[i]);

    cout << "LowPrecision::FullyConnected::BinaryInputsBinaryWeights::MultiplyInt8SingleBatch Return Status\t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    cout << "LowPrecision::FullyConnected::BinaryInputsBinaryWeights::MultiplyInt8SingleBatch\t\t=> " << ((MI8I4Passed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (MI8I4Passed?"PASSED":"FAILED") << "\033[0m" << endl;
#if PRINT_VALUES || PRINT_MUL_OUTPUT
    cout << "[";
    for (int i = 0; i < output_shape.size[0]; i++)
        cout << ((int)output_data[i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    deallocate(input_data);
    deallocate(input_data_MB);
    deallocate(input_pack_MB);
    deallocate(kernel_data);
    deallocate(kernel_data_C);
    deallocate(filter_data);
    deallocate(output_data);
    deallocate(output_data_MB);
    deallocate(output_data_R);
    deallocate(output_data_R_MB);
    cout << "LowPrecision::FullyConnected::BinaryInputsBinaryWeights  Deallocation\t\t\t\t=> \033[1m\033[32mPASSED\033[0m" << endl;
}

void run_terter_tests(
    const int* _template,
    const int8_t* _answers,
    const int kernel_fill_mode,
    const int num_inputs,
    const int num_output,
    const int num_batch
    ){
    int _input_shape[1]         = { num_inputs },
        _activation_shape[1]    = { num_inputs / 4 },
        _input_shape_MB[2]      = { num_batch , num_inputs },
        _activation_MB_shape[2] = { num_batch , num_inputs / 4 },
        _kernel_shape[2]        = { num_output, num_inputs },
        _filter_shape[2]        = { num_output, num_inputs / 4 },
        _output_shape[1]        = { num_output },
        _output_shape_MB[2]     = { num_batch , num_output };
    Shape input_shape           = get_shape(_input_shape,         1),
          activation_shape      = get_shape(_activation_shape,    1),
          input_shape_MB        = get_shape(_input_shape_MB,      2),
          activation_shape_MB   = get_shape(_activation_MB_shape, 2),
          kernel_shape          = get_shape(_kernel_shape,        2),
          filter_shape          = get_shape(_filter_shape,        2),
          output_shape          = get_shape(_output_shape,        1),
          output_shape_MB       = get_shape(_output_shape_MB,     2);
    int8_t*  input_data         = allocate<int8_t>(input_shape.flatsize);
    int8_t*  activation_data    = allocate<int8_t>(activation_shape.flatsize);
    int8_t*  input_data_MB      = allocate<int8_t>(input_shape_MB.flatsize);
    int8_t*  activation_data_MB = allocate<int8_t>(activation_shape_MB.flatsize);
    int8_t*  kernel_data        = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  kernel_data_C      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  filter_data        = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output_data        = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_MB     = allocate<int32_t>(output_shape_MB.flatsize);
    int32_t* output_data_R      = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_R_MB   = allocate<int32_t>(output_shape_MB.flatsize);

    int     input_template[]        = { -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0 };
    uint8_t activation_answers_u[]  = { 0xff, 0x55, 0x00, 0xff, 0x55, 0x00, 0xff, 0x55, 0x00, 0xff, 0x55, 0x00, 0xff, 0x55, 0x00, 0x00 };
    int8_t* activation_answers      = LowPrecision::get_pointer_as<int8_t>(activation_answers_u);
    for (int i = 0; i < num_inputs; i++)
        input_data[i] = input_template[i % 16];
    // one_vector(input_data, num_inputs);
    // half_one_half_zero_vector(input_data, num_inputs);
    
    for (int i = 0; i < input_shape_MB.size[0]; i++)
        for (int j = 0; j < input_shape_MB.size[1]; j++)
            input_data_MB[i * input_shape_MB.size[1] + j] = input_template[j % 16];
    
    if(kernel_fill_mode == 0){
        for (int i = 0; i < kernel_shape.size[0]; i++)
            for (int j = 0; j < kernel_shape.size[1]; j++)
                kernel_data[i * kernel_shape.size[1] + j]   = _template[j % 32];

        for (int i = 0; i < kernel_shape.size[1]; i++)
            for (int j = 0; j < kernel_shape.size[0]; j++)
                kernel_data_C[i * kernel_shape.size[0] + j] = _template[i % 32];
    }
    else if (kernel_fill_mode == 1){
        zero_vector(kernel_data, kernel_shape.flatsize);
        zero_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 2){
        one_vector(kernel_data, kernel_shape.flatsize);
        one_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 3){
        minus_one_vector(kernel_data, kernel_shape.flatsize);
        minus_one_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 4){
        two_vector(kernel_data, kernel_shape.flatsize);
        two_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 5){
        minus_two_vector(kernel_data, kernel_shape.flatsize);
        minus_two_vector(kernel_data_C, kernel_shape.flatsize);
    }

#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < 1; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape.size[0]; j++)
            cout << ((int)input_data[i * input_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif

    Status i_ret = LowPrecision::FullyConnected::TernaryInputsTernaryWeights::QuantizeInput(input_data, input_shape, activation_data, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::TernaryInputsTernaryWeights::QuantizeInput-RowMajor Return Status\t=> " << ((i_ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((i_ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (activation_shape.flatsize <= 128 * 128 + 128 && !i_ret){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < activation_shape.flatsize ; i++)
            QFTI4RMPassed &= (activation_data[i] == activation_answers[i % 16]);
        cout << "LowPrecision::FullyConnected::TernaryInputsTernaryWeights::QuantizeInput-RowMajor\t\t=> " << ((QFTI4RMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < 1; i++){
        cout << "\t[";
        for (int j = 0; j < activation_shape.size[0]; j++)
            cout << ((int)activation_data[i * activation_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < 1; i++){
        cout << "\t[";
        for (int j = 0; j < activation_shape.size[0]; j++)
            cout << ((((uint8_t)activation_data[i * activation_shape.size[0] + j]) < 16)?("0"):("")) << (((int)activation_data[i * activation_shape.size[0] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif

#if PRINT_VALUES || PRINT_ACTIVATIONS
    cout << LowPrecision::get_shape_string(input_shape_MB) << ", " << input_shape_MB.flatsize << endl;
    cout << "[" << endl;
    for (int i = 0; i < input_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape_MB.size[1]; j++)
            cout << ((int)input_data_MB[i * input_shape_MB.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
    i_ret = LowPrecision::FullyConnected::TernaryInputsTernaryWeights::QuantizeInput(input_data_MB, input_shape_MB, activation_data_MB, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::TernaryInputsTernaryWeights::QuantizeInput-MB-RowMajor Return Stat=> " << ((i_ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((i_ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (activation_shape_MB.flatsize <= 128 * 128 + 128 && !i_ret){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < activation_shape_MB.flatsize ; i++){
            if (activation_data_MB[i] != activation_answers[i % 16])
                cout << i << ": " << ((int)activation_data_MB[i]) << " -> " << ((int)activation_answers[i % 16]) << endl;
            QFTI4RMPassed &= (activation_data_MB[i] == activation_answers[i % 16]);
        }
        cout << "LowPrecision::FullyConnected::TernaryInputsTernaryWeights::QuantizeInput-MB-RowMajor\t\t=> " << ((QFTI4RMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_VALUES || PRINT_ACTIVATIONS
    cout << LowPrecision::get_shape_string(activation_shape_MB) << ", " << activation_shape_MB.flatsize << endl;
    cout << "[" << endl;
    for (int i = 0; i < activation_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < activation_shape_MB.size[1]; j++)
            cout << ((int)activation_data_MB[i * activation_shape_MB.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < activation_shape_MB.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < activation_shape_MB.size[1]; j++)
            cout << ((((uint8_t)activation_data_MB[i * activation_shape_MB.size[0] + j]) < 16)?("0"):("")) << (((int)activation_data_MB[i * activation_shape_MB.size[0] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif


#if PRINT_KERNEL
    cout << "Kernel = [" << endl;
    for (int i = 0; i < kernel_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < kernel_shape.size[1]; j++)
            cout << ((int)kernel_data[i * kernel_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "Kernel(0x) = [" << hex << endl;
    for (int i = 0; i < kernel_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < kernel_shape.size[1]; j++)
            cout << (((int)kernel_data[i * kernel_shape.size[1] + j]) & 0x0f) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
#if PRINT_KERNEL
    cout << "Kernel_C = [" << endl;
    for (int j = 0; j < kernel_shape.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < kernel_shape.size[1]; i++)
            cout << ((int)kernel_data_C[i * kernel_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "Kernel_C(0x) = [" << hex << endl;
    for (int j = 0; j < kernel_shape.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < kernel_shape.size[1]; i++)
            cout << (((int)kernel_data_C[i * kernel_shape.size[0] + j]) & 0x0f) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    Status ret = LowPrecision::FullyConnected::TernaryInputsTernaryWeights::QuantizeFilter(kernel_data, kernel_shape, filter_data, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::TernaryInputsTernaryWeights::QuantizeFilter-RowMajor Return Status=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 4096 && !ret){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4RMPassed &= (filter_data[i] == _answers[i]);
        cout << "LowPrecision::FullyConnected::TernaryInputsTernaryWeights::QuantizeFilter-RowMajor\t\t=> " << ((QFTI4RMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((((uint8_t)filter_data[i * filter_shape.size[1] + j]) < 16)?("0"):("")) << (((int)filter_data[i * filter_shape.size[1] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    
    ret = LowPrecision::FullyConnected::TernaryInputsTernaryWeights::QuantizeFilter(kernel_data_C, kernel_shape, filter_data, MemLayout::kColumnMajor);
    cout << "LowPrecision::FullyConnected::TernaryInputsTernaryWeights::QuantizeFilter-ColumnMajor Return Sta=> " << ((ret)?((ret == LowPrecision::Status::WrongMemLayout)?("\033[1m\033[33m"):("\033[1m\033[31m")):("\033[1m\033[32m")) << ((ret)?((ret == LowPrecision::Status::WrongMemLayout)?("NO_SUPPORT"):("FAILED")):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 4096 && !ret){
        bool QFTI4CMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4CMPassed &= (filter_data[i] == _answers[i]);
        cout << "LowPrecision::FullyConnected::TernaryInputsTernaryWeights::QuantizeFilter-ColumnMajor\t\t\t=> " << ((QFTI4CMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4CMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((((uint8_t)filter_data[i * filter_shape.size[1] + j]) < 16)?("0"):("")) << (((int)filter_data[i * filter_shape.size[1] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    minus_one_vector(output_data_R, output_shape.flatsize);

    ruy::Matrix<int8_t> ruy_lhs;
    ruy::Matrix<int8_t> ruy_rhs;
    ruy::Matrix<int8_t> ruy_rhs_C;
    ruy::Matrix<int32_t> ruy_dst;

    // Create lhs
    ruy::MakeSimpleLayout(1, input_shape.size[0], ruy::Order::kColMajor,
                              ruy_lhs.mutable_layout());
    ruy_lhs.set_data(input_data);
    // Create rhs
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kColMajor,
                              ruy_rhs.mutable_layout());
    ruy_rhs.set_data(kernel_data);
    ruy_rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create rhs
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kRowMajor,
                              ruy_rhs_C.mutable_layout());
    ruy_rhs_C.set_data(kernel_data_C);
    ruy_rhs_C.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create dst
    ruy::MakeSimpleLayout(1, output_shape.size[0], ruy::Order::kColMajor,
                              ruy_dst.mutable_layout());
    ruy_dst.set_data(output_data_R);

    ruy::Context* _ruy_context = new ruy::Context;
    ruy::MulParams<int32_t, int32_t> ruy_mul_params;

    ruy::Mul(ruy_lhs, ruy_rhs, ruy_mul_params, _ruy_context, &ruy_dst);

#if PRINT_VALUES || PRINT_MUL_OUTPUT
    cout << "[";
    for (int i = 0; i < output_shape.size[0]; i++)
        cout << ((int)output_data_R[i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    minus_one_vector(output_data, output_shape.flatsize);
    ret = LowPrecision::FullyConnected::TernaryInputsTernaryWeights::MultiplyInt8SingleBatch(
                            activation_data, input_shape,
                            filter_data, kernel_shape,
                            output_data, output_shape);
    bool MI8I4Passed = true;
    for (int i = 0 ; i < output_shape.flatsize / 2 ; i++)
        MI8I4Passed &= (output_data_R[i] == output_data[i]);

    cout << "LowPrecision::FullyConnected::TernaryInputsTernaryWeights::MultiplyInt8SingleBatch Return Status=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    cout << "LowPrecision::FullyConnected::TernaryInputsTernaryWeights::MultiplyInt8SingleBatch\t\t=> " << ((MI8I4Passed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (MI8I4Passed?"PASSED":"FAILED") << "\033[0m" << endl;
#if PRINT_VALUES || PRINT_MUL_OUTPUT
    cout << "[";
    for (int i = 0; i < output_shape.size[0]; i++)
        cout << ((int)output_data[i]) << ", ";
    cout << "]";
    cout << endl;
#endif
    
    zero_vector(output_data_R_MB, output_shape_MB.flatsize);

    ruy::Matrix<int8_t> ruy_MB_lhs;
    ruy::Matrix<int8_t> ruy_MB_rhs;
    ruy::Matrix<int8_t> ruy_MB_rhs_C;
    ruy::Matrix<int32_t> ruy_MB_dst;

    // Create lhs
    ruy::MakeSimpleLayout(input_shape_MB.size[0], input_shape_MB.size[1], ruy::Order::kRowMajor,
                              ruy_MB_lhs.mutable_layout());
    ruy_MB_lhs.set_data(input_data_MB);
    // Create rhs (Column Major)
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kColMajor,
                              ruy_MB_rhs.mutable_layout());
    ruy_MB_rhs.set_data(kernel_data);
    ruy_MB_rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create rhs (Row Major)
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kRowMajor,
                              ruy_MB_rhs_C.mutable_layout());
    ruy_MB_rhs_C.set_data(kernel_data_C);
    ruy_MB_rhs_C.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create dst
    ruy::MakeSimpleLayout(output_shape_MB.size[0], output_shape_MB.size[1], ruy::Order::kColMajor,
                              ruy_MB_dst.mutable_layout());
    ruy_MB_dst.set_data(output_data_R_MB);

    ruy::Context* _ruy_MB_context = new ruy::Context;
    ruy::MulParams<int32_t, int32_t> ruy_MB_mul_params;

    ruy::Mul(ruy_MB_lhs, ruy_MB_rhs, ruy_MB_mul_params, _ruy_MB_context, &ruy_MB_dst);

#if PRINT_VALUES || PRINT_MUL_OUTPUT || PRINT_MUL_MB_OUTPUT
    cout << "[";
    for (int j = 0; j < output_shape_MB.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < output_shape_MB.size[1]; i++)
            cout << ((int)output_data_R_MB[j * output_shape_MB.size[1] + i]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif

    zero_vector(output_data_MB, output_shape_MB.flatsize);
    ret = LowPrecision::FullyConnected::TernaryInputsTernaryWeights::MultiplyInt8MultiBatched(
                            activation_data_MB, input_shape_MB,
                            filter_data, kernel_shape,
                            output_data_MB, output_shape_MB);
    bool MI8I4MBPassed = true;
    for (int i = 0 ; i < output_shape_MB.flatsize ; i++)
        MI8I4MBPassed &= (output_data_R_MB[i] == output_data_MB[i]);
    cout << "LowPrecision::FullyConnected::TernaryInputsTernaryWeights::MultiplyInt8MultiBatched Return Statu=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    cout << "LowPrecision::FullyConnected::TernaryInputsTernaryWeights::MultiplyInt8MultiBatched\t\t=> " << ((MI8I4MBPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (MI8I4MBPassed?"PASSED":"FAILED") << "\033[0m" << endl;

#if PRINT_VALUES || PRINT_MUL_OUTPUT || PRINT_MUL_MB_OUTPUT
    cout << "[";
    for (int j = 0; j < output_shape_MB.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < output_shape_MB.size[1]; i++)
            cout << ((int)output_data_MB[j * output_shape_MB.size[1] + i]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif

    deallocate(input_data);
    deallocate(input_data_MB);
    deallocate(activation_data_MB);
    deallocate(kernel_data);
    deallocate(kernel_data_C);
    deallocate(filter_data);
    deallocate(output_data);
    deallocate(output_data_MB);
    deallocate(output_data_R);
    deallocate(output_data_R_MB);
    cout << "LowPrecision::FullyConnected::TernaryInputsTernaryWeights  Deallocation\t\t\t\t=> \033[1m\033[32mPASSED\033[0m" << endl;
}

void run_i3i3_tests(
    const int* _template,
    const int8_t* _answers,
    const int kernel_fill_mode,
    const int num_inputs,
    const int num_output,
    const int num_batch
    ){
    int _input_shape[1]       = { num_inputs },
        _activation_shape[1]  = { (num_inputs / 5) * 2 },
        _input_shape_MB[2]    = { num_batch , num_inputs },
        _kernel_shape[2]      = { num_output, num_inputs },
        _filter_shape[2]      = { num_output, (num_inputs / 5) * 2 },
        _output_shape[1]      = { num_output },
        _output_shape_MB[2]   = { num_batch , num_output };
    Shape input_shape         = get_shape(_input_shape,      1),
          activation_shape    = get_shape(_activation_shape, 1),
          input_shape_MB      = get_shape(_input_shape_MB,   2),
          kernel_shape        = get_shape(_kernel_shape,     2),
          filter_shape        = get_shape(_filter_shape,     2),
          output_shape        = get_shape(_output_shape,     1),
          output_shape_MB     = get_shape(_output_shape_MB,  2);
    
    int8_t*  input_data       = allocate<int8_t>(input_shape.flatsize);
    int8_t*  activation_data  = allocate<int8_t>(activation_shape.flatsize);
    int8_t*  input_data_MB    = allocate<int8_t>(input_shape_MB.flatsize);
    int8_t*  input_pack_MB    = allocate<int8_t>(input_shape_MB.flatsize);
    int8_t*  kernel_data      = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  kernel_data_C    = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  filter_data      = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output_data      = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_MB   = allocate<int32_t>(output_shape_MB.flatsize);
    int32_t* output_data_R    = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_R_MB = allocate<int32_t>(output_shape_MB.flatsize);

    int     input_template[]        = { -2, -1, 0, 1, 2, -1, 0, 1 };
    uint8_t activation_answers_u[]  = { 0x92, 0x24, 0x92, 0x24, 0x92, 0x24, 0x92, 0x24, 0x92, 0x24, 0x92, 0x24, 0x92, 0x24, 0x92, 0x24 };
    int8_t* activation_answers      = LowPrecision::get_pointer_as<int8_t>(activation_answers_u);
    for (int i = 0; i < num_inputs; i++)
        input_data[i] = input_template[i % 8];
    one_vector(input_data, num_inputs);
    // half_one_half_zero_vector(input_data, num_inputs);
    
    for (int i = 0; i < num_batch; i++)
        for (int j = 0; j < num_inputs; j++)
            input_data_MB[i * num_inputs + j] = 1;
    
    if(kernel_fill_mode == 0){
        for (int i = 0; i < kernel_shape.size[0]; i++)
            for (int j = 0; j < kernel_shape.size[1]; j++)
                kernel_data[i * kernel_shape.size[1] + j]   = _template[j % 32];

        for (int i = 0; i < kernel_shape.size[1]; i++)
            for (int j = 0; j < kernel_shape.size[0]; j++)
                kernel_data_C[i * kernel_shape.size[0] + j] = _template[i % 32];
    }
    else if (kernel_fill_mode == 1){
        zero_vector(kernel_data, kernel_shape.flatsize);
        zero_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 2){
        one_vector(kernel_data, kernel_shape.flatsize);
        one_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 3){
        minus_one_vector(kernel_data, kernel_shape.flatsize);
        minus_one_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 4){
        two_vector(kernel_data, kernel_shape.flatsize);
        two_vector(kernel_data_C, kernel_shape.flatsize);
    }
    else if (kernel_fill_mode == 5){
        minus_two_vector(kernel_data, kernel_shape.flatsize);
        minus_two_vector(kernel_data_C, kernel_shape.flatsize);
    }

#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < 1; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape.size[0]; j++)
            cout << ((int)input_data[i * input_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < 1; i++){
        cout << "\t[";
        for (int j = 0; j < input_shape.size[0]; j++)
            cout << ((((uint8_t)input_data[i * input_shape.size[0] + j]) < 16)?("0"):("")) << (((int)input_data[i * input_shape.size[0] + j]) & 0xff) << ", ";
            // cout << ((int)input_data[i * input_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif

    Status i_ret = LowPrecision::FullyConnected::Int3InputsInt3Weights::QuantizeInput(input_data, input_shape, activation_data, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::Int3InputsInt3Weights::QuantizeInput-RowMajor Return Status\t=> " << ((i_ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((i_ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (activation_shape.flatsize <= 128 * 128 + 128 && !i_ret){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < activation_shape.flatsize ; i++)
            QFTI4RMPassed &= (activation_data[i] == activation_answers[i % 16]);
        cout << "LowPrecision::FullyConnected::Int3InputsInt3Weights::QuantizeInput-RowMajor\t\t\t=> " << ((QFTI4RMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_VALUES
    cout << "[" << endl;
    for (int i = 0; i < 1; i++){
        cout << "\t[";
        for (int j = 0; j < activation_shape.size[0]; j++)
            cout << ((int)activation_data[i * activation_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_VALUES_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < 1; i++){
        cout << "\t[";
        for (int j = 0; j < activation_shape.size[0]; j++)
            cout << ((((uint8_t)activation_data[i * activation_shape.size[0] + j]) < 16)?("0"):("")) << (((int)activation_data[i * activation_shape.size[0] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif

#if PRINT_KERNEL
    cout << "Kernel = [" << endl;
    for (int i = 0; i < kernel_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < kernel_shape.size[1]; j++)
            cout << ((int)kernel_data[i * kernel_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_KERNEL_IN_HEX
    cout << "Kernel(0x) = [" << hex << endl;
    for (int i = 0; i < kernel_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < kernel_shape.size[1]; j++)
            cout << (((int)kernel_data[i * kernel_shape.size[1] + j]) & 0x0f) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
#if PRINT_KERNEL
    cout << "Kernel_C = [" << endl;
    for (int j = 0; j < kernel_shape.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < kernel_shape.size[1]; i++)
            cout << ((int)kernel_data_C[i * kernel_shape.size[0] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_KERNEL_IN_HEX
    cout << "Kernel_C(0x) = [" << hex << endl;
    for (int j = 0; j < kernel_shape.size[0]; j++){
        cout << "\t[";
        for (int i = 0; i < kernel_shape.size[1]; i++)
            cout << (((int)kernel_data_C[i * kernel_shape.size[0] + j]) & 0x0f) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    Status ret = LowPrecision::FullyConnected::Int3InputsInt3Weights::QuantizeFilter(kernel_data, kernel_shape, filter_data, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::Int3InputsInt3Weights::QuantizeFilter-RowMajor Return Status\t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 4096 && !ret){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4RMPassed &= (filter_data[i] == _answers[i]);
        cout << "LowPrecision::FullyConnected::Int3InputsInt3Weights::QuantizeFilter-RowMajor\t\t\t=> " << ((QFTI4RMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_FILTER
    cout << "[" << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
        cout << "]," << endl;
    }
    cout << "]";
    cout << endl;
#endif
#if PRINT_FILTER_IN_HEX
    cout << "(0x)[" << hex << endl;
    for (int i = 0; i < filter_shape.size[0]; i++){
        cout << "\t[";
        for (int j = 0; j < filter_shape.size[1]; j++)
            cout << ((((uint8_t)filter_data[i * filter_shape.size[1] + j]) < 16)?("0"):("")) << (((int)filter_data[i * filter_shape.size[1] + j]) & 0xff) << ", ";
        cout << "]," << endl;
    }
    cout << dec << "]";
    cout << endl;
#endif
    
    ret = LowPrecision::FullyConnected::Int3InputsInt3Weights::QuantizeFilter(kernel_data_C, kernel_shape, filter_data, MemLayout::kColumnMajor);
    cout << "LowPrecision::FullyConnected::Int3InputsInt3Weights::QuantizeFilter-ColumnMajor Return Status\t=> " << ((ret)?((ret == LowPrecision::Status::WrongMemLayout)?("\033[1m\033[33m"):("\033[1m\033[31m")):("\033[1m\033[32m")) << ((ret)?((ret == LowPrecision::Status::WrongMemLayout)?("NO_SUPPORT"):("FAILED")):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 4096 && !ret){
        bool QFTI4CMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4CMPassed &= (filter_data[i] == _answers[i]);
        cout << "LowPrecision::FullyConnected::Int3InputsInt3Weights::QuantizeFilter-ColumnMajor\t\t\t\t=> " << ((QFTI4CMPassed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (QFTI4CMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
    }
#if PRINT_FILTER
    if (!ret){
        cout << "[" << endl;
        for (int i = 0; i < filter_shape.size[0]; i++){
            cout << "\t[";
            for (int j = 0; j < filter_shape.size[1]; j++)
                cout << ((int)filter_data[i * filter_shape.size[1] + j]) << ", ";
            cout << "]," << endl;
        }
        cout << "]";
        cout << endl;
    }
#endif
#if PRINT_FILTER_IN_HEX
    if (!ret){
        cout << "(0x)[" << hex << endl;
        for (int i = 0; i < filter_shape.size[0]; i++){
            cout << "\t[";
            for (int j = 0; j < filter_shape.size[1]; j++)
                cout << ((((uint8_t)filter_data[i * filter_shape.size[1] + j]) < 16)?("0"):("")) << (((int)filter_data[i * filter_shape.size[1] + j]) & 0xff) << ", ";
            cout << "]," << endl;
        }
        cout << dec << "]";
        cout << endl;
    }
#endif
    minus_one_vector(output_data_R, output_shape.flatsize);

    ruy::Matrix<int8_t> ruy_lhs;
    ruy::Matrix<int8_t> ruy_rhs;
    ruy::Matrix<int8_t> ruy_rhs_C;
    ruy::Matrix<int32_t> ruy_dst;

    // Create lhs
    ruy::MakeSimpleLayout(1, input_shape.size[0], ruy::Order::kColMajor,
                              ruy_lhs.mutable_layout());
    ruy_lhs.set_data(input_data);
    // Create rhs
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kColMajor,
                              ruy_rhs.mutable_layout());
    ruy_rhs.set_data(kernel_data);
    ruy_rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create rhs
    ruy::MakeSimpleLayout(kernel_shape.size[0], kernel_shape.size[1], ruy::Order::kRowMajor,
                              ruy_rhs_C.mutable_layout());
    ruy_rhs_C.set_data(kernel_data_C);
    ruy_rhs_C.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create dst
    ruy::MakeSimpleLayout(1, output_shape.size[0], ruy::Order::kColMajor,
                              ruy_dst.mutable_layout());
    ruy_dst.set_data(output_data_R);

    ruy::Context* _ruy_context = new ruy::Context;
    ruy::MulParams<int32_t, int32_t> ruy_mul_params;

    ruy::Mul(ruy_lhs, ruy_rhs, ruy_mul_params, _ruy_context, &ruy_dst);

#if PRINT_VALUES || PRINT_MUL_OUTPUT
    cout << "[";
    for (int i = 0; i < output_shape.size[0]; i++)
        cout << ((int)output_data_R[i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    minus_one_vector(output_data, output_shape.flatsize);
    ret = LowPrecision::FullyConnected::Int3InputsInt3Weights::MultiplyInt8SingleBatch(
                            activation_data, input_shape,
                            filter_data, kernel_shape,
                            output_data, output_shape);
    bool MI8I4Passed = true;
    for (int i = 0 ; i < output_shape.flatsize / 2 ; i++)
        MI8I4Passed &= (output_data_R[i] == output_data[i]);

    cout << "LowPrecision::FullyConnected::Int3InputsInt3Weights::MultiplyInt8SingleBatch Return Status\t=> " << ((ret)?("\033[1m\033[31m"):("\033[1m\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    cout << "LowPrecision::FullyConnected::Int3InputsInt3Weights::MultiplyInt8SingleBatch\t\t\t=> " << ((MI8I4Passed)?("\033[1m\033[32m"):("\033[1m\033[31m")) << (MI8I4Passed?"PASSED":"FAILED") << "\033[0m" << endl;
#if PRINT_VALUES || PRINT_MUL_OUTPUT
    cout << "[";
    for (int i = 0; i < output_shape.size[0]; i++)
        cout << ((int)output_data[i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    deallocate(input_data);
    deallocate(input_data_MB);
    deallocate(input_pack_MB);
    deallocate(kernel_data);
    deallocate(kernel_data_C);
    deallocate(filter_data);
    deallocate(output_data);
    deallocate(output_data_MB);
    deallocate(output_data_R);
    deallocate(output_data_R_MB);
    cout << "LowPrecision::FullyConnected::Int3InputsInt3Weights  Deallocation\t\t\t\t=> \033[1m\033[32mPASSED\033[0m" << endl;
}

int main(int argc, char *argv[]){
    std::string input_mode = "";
    if (argc >= 2)
        input_mode = argv[1];
    bool singlebatch_benchmark_enable = false;
    bool multibatch_benchmark_enable = false;
    bool integrity_test = true;

    int  selected_test = 0x0040;
    int  selected_benchmark_enable = 0xffff;
    int  benchmark_iterations = 2000;
    int  test_mul_api = 0x0000;
    int  selected_benchmark_real_mul_api = 0x0000;
    int  selected_benchmark_real_single_mul_api = 0x0000;
    int  selected_benchmark_real_multi_mul_api = 0x0000;
    int  enable_single_mul_api_increasing_size_benchmark = 0x0000;
    int  enable_single_mul_api_different_size_benchmark = 0x0000;
    int  enable_multi_mul_api_different_size_benchmark = 0x0000;

    std::string single_mul_api_different_size_benchmark_time_file = "/data/local/tmp/single-mul-api-different-size-time.csv";
    std::string single_mul_api_different_size_benchmark_speedup_file = "/data/local/tmp/single-mul-api-different-size-speedup.csv";
    std::string multi_mul_api_different_size_benchmark_time_file = "/data/local/tmp/multi-mul-api-different-size-time.csv";
    std::string multi_mul_api_different_size_benchmark_speedup_file = "/data/local/tmp/multi-mul-api-different-size-speedup.csv";

    if (LowPrecision::FullyConnected::GetVariableFromEnv( "BenchmarkIterations" ) != "")
        benchmark_iterations = std::stoi(LowPrecision::FullyConnected::GetVariableFromEnv( "BenchmarkIterations" ));
    if (input_mode == "benchmark"){
        singlebatch_benchmark_enable = true;
        multibatch_benchmark_enable = true;
        integrity_test = false;
    }
    else if (input_mode == "benchmark-single-batch"){
        singlebatch_benchmark_enable = true;
        multibatch_benchmark_enable = false;
        integrity_test = false;
        if (argc >= 3){
            std::string selected_benchmark = "";
            selected_benchmark = argv[2];
            if (selected_benchmark == "All")
                selected_benchmark_enable = 0xffff; 
            else if (selected_benchmark == "Int4")
                selected_benchmark_enable = 0x0001; 
            else if (selected_benchmark == "Binary")
                selected_benchmark_enable = 0x0002; 
            else if (selected_benchmark == "Ternary")
                selected_benchmark_enable = 0x0004; 
            else if (selected_benchmark == "Quaternary")
                selected_benchmark_enable = 0x0008; 
            else if (selected_benchmark == "Int4InputsInt8Weights")
                selected_benchmark_enable = 0x0010; 
            else if (selected_benchmark == "Int4InputsInt4Weights")
                selected_benchmark_enable = 0x0020; 
            else if (selected_benchmark == "BinaryInputsInt8Weights")
                selected_benchmark_enable = 0x0080; 
            else if (selected_benchmark == "BinaryInputsBinaryWeights")
                selected_benchmark_enable = 0x0100; 
            else if (selected_benchmark == "TernaryInputsInt8Weights")
                selected_benchmark_enable = 0x0040; 
            else if (selected_benchmark == "TernaryInputsTernaryWeights")
                selected_benchmark_enable = 0x0200; 
            // else if (selected_benchmark == "Int3InputsInt3Weights")
            //     selected_benchmark_enable = 0x0400; 
            else if (selected_benchmark == "Int8")
                selected_benchmark_enable = 0x8000; 
        }
    }
    else if (input_mode == "benchmark-multi-batch"){
        singlebatch_benchmark_enable = false;
        multibatch_benchmark_enable = true;
        integrity_test = false;
        if (argc >= 3){
            std::string selected_benchmark = "";
            selected_benchmark = argv[2];
            if (selected_benchmark == "All")
                selected_benchmark_enable = 0x7fff; 
            else if (selected_benchmark == "Int4")
                selected_benchmark_enable = 0x0001; 
            else if (selected_benchmark == "Binary")
                selected_benchmark_enable = 0x0002; 
            else if (selected_benchmark == "Ternary")
                selected_benchmark_enable = 0x0004; 
            else if (selected_benchmark == "Quaternary")
                selected_benchmark_enable = 0x0008; 
            else if (selected_benchmark == "Int4InputsInt8Weights")
                selected_benchmark_enable = 0x0010; 
            else if (selected_benchmark == "Int4InputsInt4Weights")
                selected_benchmark_enable = 0x0020; 
            else if (selected_benchmark == "BinaryInputsInt8Weights")
                selected_benchmark_enable = 0x0080; 
            else if (selected_benchmark == "BinaryInputsBinaryWeights")
                selected_benchmark_enable = 0x0100; 
            else if (selected_benchmark == "TernaryInputsInt8Weights")
                selected_benchmark_enable = 0x0040; 
            else if (selected_benchmark == "TernaryInputsTernaryWeights")
                selected_benchmark_enable = 0x0200; 
            // else if (selected_benchmark == "Int3InputsInt3Weights")
            //     selected_benchmark_enable = 0x0400; 
            else if (selected_benchmark == "Int8")
                selected_benchmark_enable = 0x8000; 
        }
    }
    else if (input_mode == "test-mul-api"){
        singlebatch_benchmark_enable = false;
        multibatch_benchmark_enable = false;
        integrity_test = false;
        if (argc >= 3){
            std::string selected_test = "";
            selected_test = argv[2];
            if (selected_test == "All")
                test_mul_api = 0xffff; 
            else if (selected_test == "Int4")
                test_mul_api = 0x0001; 
            else if (selected_test == "Binary")
                test_mul_api = 0x0002; 
            else if (selected_test == "Ternary")
                test_mul_api = 0x0004; 
            else if (selected_test == "Quaternary")
                test_mul_api = 0x0008; 
            else if (selected_test == "Int4InputsInt8Weights")
                test_mul_api = 0x0010; 
            else if (selected_test == "Int4InputsInt4Weights")
                test_mul_api = 0x0020; 
            else if (selected_test == "BinaryInputsInt8Weights")
                test_mul_api = 0x0080; 
            else if (selected_test == "BinaryInputsBinaryWeights")
                test_mul_api = 0x0100; 
            else if (selected_test == "TernaryInputsInt8Weights")
                test_mul_api = 0x0040; 
            else if (selected_test == "TernaryInputsTernaryWeights")
                test_mul_api = 0x0200; 
            // else if (selected_test == "Int3InputsInt3Weights")
            //     test_mul_api = 0x0400;
        }
    }
    else if (input_mode == "benchmark-real-mul-api"){
        singlebatch_benchmark_enable = false;
        multibatch_benchmark_enable = false;
        integrity_test = false;
        if (argc >= 3){
            std::string selected_test = "";
            selected_test = argv[2];
            if (selected_test == "All")
                selected_benchmark_real_mul_api = 0xffff; 
            else if (selected_test == "Int4")
                selected_benchmark_real_mul_api = 0x0001; 
            else if (selected_test == "Binary")
                selected_benchmark_real_mul_api = 0x0002; 
            else if (selected_test == "Ternary")
                selected_benchmark_real_mul_api = 0x0004; 
            else if (selected_test == "Quaternary")
                selected_benchmark_real_mul_api = 0x0008; 
            else if (selected_test == "Int4InputsInt8Weights")
                selected_benchmark_real_mul_api = 0x0010; 
            else if (selected_test == "Int4InputsInt4Weights")
                selected_benchmark_real_mul_api = 0x0020; 
            else if (selected_test == "BinaryInputsInt8Weights")
                selected_benchmark_real_mul_api = 0x0080; 
            else if (selected_test == "BinaryInputsBinaryWeights")
                selected_benchmark_real_mul_api = 0x0100; 
            else if (selected_test == "TernaryInputsInt8Weights")
                selected_benchmark_real_mul_api = 0x0040; 
            else if (selected_test == "TernaryInputsTernaryWeights")
                selected_benchmark_real_mul_api = 0x0200; 
            // else if (selected_test == "Int3InputsInt3Weights")
            //     selected_benchmark_real_mul_api = 0x0400;
            else if (selected_test == "Int8")
                selected_benchmark_real_mul_api = 0x8000;
        }
    }
    else if (input_mode == "benchmark-real-single-mul-api"){
        singlebatch_benchmark_enable = false;
        multibatch_benchmark_enable = false;
        integrity_test = false;
        if (argc >= 3){
            std::string selected_test = "";
            selected_test = argv[2];
            if (selected_test == "All")
                selected_benchmark_real_single_mul_api = 0xffff; 
            else if (selected_test == "Int4")
                selected_benchmark_real_single_mul_api = 0x0001; 
            else if (selected_test == "Binary")
                selected_benchmark_real_single_mul_api = 0x0002; 
            else if (selected_test == "Ternary")
                selected_benchmark_real_single_mul_api = 0x0004; 
            else if (selected_test == "Quaternary")
                selected_benchmark_real_single_mul_api = 0x0008; 
            else if (selected_test == "Int4InputsInt8Weights")
                selected_benchmark_real_single_mul_api = 0x0010; 
            else if (selected_test == "Int4InputsInt4Weights")
                selected_benchmark_real_single_mul_api = 0x0020; 
            else if (selected_test == "BinaryInputsInt8Weights")
                selected_benchmark_real_single_mul_api = 0x0080; 
            else if (selected_test == "BinaryInputsBinaryWeights")
                selected_benchmark_real_single_mul_api = 0x0100; 
            else if (selected_test == "TernaryInputsInt8Weights")
                selected_benchmark_real_single_mul_api = 0x0040; 
            else if (selected_test == "TernaryInputsTernaryWeights")
                selected_benchmark_real_single_mul_api = 0x0200; 
            // else if (selected_test == "Int3InputsInt3Weights")
            //     selected_benchmark_real_single_mul_api = 0x0400;
            else if (selected_test == "Int8")
                selected_benchmark_real_single_mul_api = 0x8000;
        }
        else
            selected_benchmark_real_single_mul_api = 0xffff;
    }
    else if (input_mode == "benchmark-real-multi-mul-api"){
        singlebatch_benchmark_enable = false;
        multibatch_benchmark_enable = false;
        integrity_test = false;
        if (argc >= 3){
            std::string selected_test = "";
            selected_test = argv[2];
            if (selected_test == "All")
                selected_benchmark_real_multi_mul_api = 0xffff; 
            else if (selected_test == "Int4")
                selected_benchmark_real_multi_mul_api = 0x0001; 
            else if (selected_test == "Binary")
                selected_benchmark_real_multi_mul_api = 0x0002; 
            else if (selected_test == "Ternary")
                selected_benchmark_real_multi_mul_api = 0x0004; 
            else if (selected_test == "Quaternary")
                selected_benchmark_real_multi_mul_api = 0x0008; 
            else if (selected_test == "Int4InputsInt8Weights")
                selected_benchmark_real_multi_mul_api = 0x0010; 
            else if (selected_test == "Int4InputsInt4Weights")
                selected_benchmark_real_multi_mul_api = 0x0020; 
            else if (selected_test == "BinaryInputsInt8Weights")
                selected_benchmark_real_multi_mul_api = 0x0080; 
            else if (selected_test == "BinaryInputsBinaryWeights")
                selected_benchmark_real_multi_mul_api = 0x0100; 
            else if (selected_test == "TernaryInputsInt8Weights")
                selected_benchmark_real_multi_mul_api = 0x0040; 
            else if (selected_test == "TernaryInputsTernaryWeights")
                selected_benchmark_real_multi_mul_api = 0x0200; 
            // else if (selected_test == "Int3InputsInt3Weights")
            //     selected_benchmark_real_multi_mul_api = 0x0400;
            else if (selected_test == "Int8")
                selected_benchmark_real_multi_mul_api = 0x8000;
        }
        else
            selected_benchmark_real_multi_mul_api = 0xffff;
    }
    else if (input_mode == "benchmark-single-mul-api-increasing-size"){
        singlebatch_benchmark_enable = false;
        multibatch_benchmark_enable = false;
        integrity_test = false;
        if (argc >= 3){
            std::string selected_test = "";
            selected_test = argv[2];
            if (selected_test == "All")
                enable_single_mul_api_increasing_size_benchmark = 0xffff; 
            else if (selected_test == "Int4")
                enable_single_mul_api_increasing_size_benchmark = 0x0001; 
            else if (selected_test == "Binary")
                enable_single_mul_api_increasing_size_benchmark = 0x0002; 
            else if (selected_test == "Ternary")
                enable_single_mul_api_increasing_size_benchmark = 0x0004; 
            else if (selected_test == "Quaternary")
                enable_single_mul_api_increasing_size_benchmark = 0x0008; 
            else if (selected_test == "Int4InputsInt8Weights")
                enable_single_mul_api_increasing_size_benchmark = 0x0010; 
            else if (selected_test == "Int4InputsInt4Weights")
                enable_single_mul_api_increasing_size_benchmark = 0x0020; 
            else if (selected_test == "BinaryInputsInt8Weights")
                enable_single_mul_api_increasing_size_benchmark = 0x0080; 
            else if (selected_test == "BinaryInputsBinaryWeights")
                enable_single_mul_api_increasing_size_benchmark = 0x0100; 
            else if (selected_test == "TernaryInputsInt8Weights")
                enable_single_mul_api_increasing_size_benchmark = 0x0040; 
            else if (selected_test == "TernaryInputsTernaryWeights")
                enable_single_mul_api_increasing_size_benchmark = 0x0200; 
            // else if (selected_test == "Int3InputsInt3Weights")
            //     enable_single_mul_api_increasing_size_benchmark = 0x0400;
            else if (selected_test == "Int8")
                enable_single_mul_api_increasing_size_benchmark = 0x8000;
        }
    }
    else if (input_mode == "benchmark-single-mul-api-different-size"){
        integrity_test = false;
        if (argc >= 3){
            std::string selected_test = "";
            selected_test = argv[2];
            if (selected_test == "All")
                enable_single_mul_api_different_size_benchmark = 0xffff; 
            else if (selected_test == "Int4")
                enable_single_mul_api_different_size_benchmark = 0x0001; 
            else if (selected_test == "Binary")
                enable_single_mul_api_different_size_benchmark = 0x0002; 
            else if (selected_test == "Ternary")
                enable_single_mul_api_different_size_benchmark = 0x0004; 
            else if (selected_test == "Quaternary")
                enable_single_mul_api_different_size_benchmark = 0x0008; 
            else if (selected_test == "Int4InputsInt8Weights")
                enable_single_mul_api_different_size_benchmark = 0x0010; 
            else if (selected_test == "Int4InputsInt4Weights")
                enable_single_mul_api_different_size_benchmark = 0x0020; 
            else if (selected_test == "BinaryInputsInt8Weights")
                enable_single_mul_api_different_size_benchmark = 0x0080; 
            else if (selected_test == "BinaryInputsBinaryWeights")
                enable_single_mul_api_different_size_benchmark = 0x0100; 
            else if (selected_test == "TernaryInputsInt8Weights")
                enable_single_mul_api_different_size_benchmark = 0x0040; 
            else if (selected_test == "TernaryInputsTernaryWeights")
                enable_single_mul_api_different_size_benchmark = 0x0200; 
            // else if (selected_test == "Int3InputsInt3Weights")
            //     enable_single_mul_api_different_size_benchmark = 0x0400;
            else if (selected_test == "Int8")
                enable_single_mul_api_different_size_benchmark = 0x8000;
        }
        else enable_multi_mul_api_different_size_benchmark = 0xffff;
        if (argc >= 4){
            multi_mul_api_different_size_benchmark_time_file = string(argv[3]) + "-time.csv";
            multi_mul_api_different_size_benchmark_speedup_file = string(argv[3]) + "-speedup.csv";
        }
        if (argc >= 5){
            multi_mul_api_different_size_benchmark_time_file = string(argv[3]);
            multi_mul_api_different_size_benchmark_speedup_file = string(argv[4]);
        }
    }
    else if (input_mode == "benchmark-multi-mul-api-different-size"){
        integrity_test = false;
        if (argc >= 3){
            std::string selected_test = "";
            selected_test = argv[2];
            if (selected_test == "All")
                enable_multi_mul_api_different_size_benchmark = 0xffff; 
            else if (selected_test == "Int4")
                enable_multi_mul_api_different_size_benchmark = 0x0001; 
            else if (selected_test == "Binary")
                enable_multi_mul_api_different_size_benchmark = 0x0002; 
            else if (selected_test == "Ternary")
                enable_multi_mul_api_different_size_benchmark = 0x0004; 
            else if (selected_test == "Quaternary")
                enable_multi_mul_api_different_size_benchmark = 0x0008; 
            else if (selected_test == "Int4InputsInt8Weights")
                enable_multi_mul_api_different_size_benchmark = 0x0010; 
            else if (selected_test == "Int4InputsInt4Weights")
                enable_multi_mul_api_different_size_benchmark = 0x0020; 
            else if (selected_test == "BinaryInputsInt8Weights")
                enable_multi_mul_api_different_size_benchmark = 0x0080; 
            else if (selected_test == "BinaryInputsBinaryWeights")
                enable_multi_mul_api_different_size_benchmark = 0x0100; 
            else if (selected_test == "TernaryInputsInt8Weights")
                enable_multi_mul_api_different_size_benchmark = 0x0040; 
            else if (selected_test == "TernaryInputsTernaryWeights")
                enable_multi_mul_api_different_size_benchmark = 0x0200; 
            // else if (selected_test == "Int3InputsInt3Weights")
            //     enable_multi_mul_api_different_size_benchmark = 0x0400;
            else if (selected_test == "Int8")
                enable_multi_mul_api_different_size_benchmark = 0x8000;
        }
        else enable_multi_mul_api_different_size_benchmark = 0xffff;
        if (argc >= 4){
            multi_mul_api_different_size_benchmark_time_file = string(argv[3]) + "-time.csv";
            multi_mul_api_different_size_benchmark_speedup_file = string(argv[3]) + "-speedup.csv";
        }
        if (argc >= 5){
            multi_mul_api_different_size_benchmark_time_file = string(argv[3]);
            multi_mul_api_different_size_benchmark_speedup_file = string(argv[4]);
        }
    }
    else{
        singlebatch_benchmark_enable = false;
        multibatch_benchmark_enable = false;
        integrity_test = true;
        if (!input_mode.empty()){
            if (input_mode == "All")
                selected_test = 0xffff; 
            else if (input_mode == "Int4")
                selected_test = 0x0001; 
            else if (input_mode == "Binary")
                selected_test = 0x0002; 
            else if (input_mode == "Ternary")
                selected_test = 0x0004; 
            else if (input_mode == "Quaternary")
                selected_test = 0x0008; 
            else if (input_mode == "Int4InputsInt8Weights")
                selected_test = 0x0010; 
            else if (input_mode == "Int4InputsInt4Weights")
                selected_test = 0x0020; 
            else if (input_mode == "BinaryInputsInt8Weights")
                selected_test = 0x0080; 
            else if (input_mode == "BinaryInputsBinaryWeights")
                selected_test = 0x0100; 
            else if (input_mode == "TernaryInputsInt8Weights")
                selected_test = 0x0040; 
            else if (input_mode == "TernaryInputsTernaryWeights")
                selected_test = 0x0200; 
            else if (input_mode == "Int3InputsInt3Weights")
                selected_test = 0x0400; 
            else if (input_mode == "Int8")
                selected_test = 0x8000; 
        }
    }
    
    int mode = (integrity_test)?(selected_test):(0x0000);

    if (mode & 0x0001)/* Int4 */{
        const int i8i4_template[] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
        int i8i4_kernel_fill_mode = 0;
        const int8_t i8i4_answers_no_pad[] = {
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79
        };
        const int8_t i8i4_answers_pad[] = {
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79,
            11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        };
        const int   i8i4_num_inputs    = 32,
                    i8i4_num_output    = 32,
                    i8i4_num_batch     = 4;
        const int8_t* i8i4_answers = (i8i4_num_inputs % 32)?(i8i4_answers_pad):(i8i4_answers_no_pad); 
        run_i8i4_tests(i8i4_template, i8i4_answers, i8i4_kernel_fill_mode, i8i4_num_inputs, i8i4_num_output, i8i4_num_batch);
    }
    if (mode & 0x0002)/* Binary */{
        const int i8bin_template[] = {
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 
        };
        int i8bin_kernel_fill_mode = 0;
        const int8_t i8bin_answers[] = {
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0
        };
        const int   i8bin_num_inputs    = 128,
                    i8bin_num_output    = 128,
                    i8bin_num_batch     = 4;
        run_i8bin_tests(i8bin_template, i8bin_answers, i8bin_kernel_fill_mode, i8bin_num_inputs, i8bin_num_output, i8bin_num_batch);
    }
    if (mode & 0x0004)/* Ternary */{
        const int i8ter_template[] = {
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 
        };
        int i8ter_kernel_fill_mode = 0;
        const int8_t i8ter_answers[] = {
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
        };
        const int   i8ter_num_inputs    = 64,
                    i8ter_num_output    = 64,
                    i8ter_num_batch     = 4;
        run_i8ter_tests(i8ter_template, i8ter_answers, i8ter_kernel_fill_mode, i8ter_num_inputs, i8ter_num_output, i8ter_num_batch);
    }
    if (mode & 0x0008)/* Quaternary */{
        const int i8qua_template[] = {
            -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2, 2, 
            -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2, 2, 
            -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2, 2, 
            -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2, 2, 
            -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2, 2, 
            -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2, 2, 
            -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2, 2, 
            -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2, 2, 
        };
        int i8qua_kernel_fill_mode = 0;
        const int8_t i8qua_answers[] = {
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
            -86, 0, -1, 85, -86, 0, -1, 85,
        };
        const int   i8qua_num_inputs    = 64,
                    i8qua_num_output    = 64,
                    i8qua_num_batch     = 4;
        run_i8qua_tests(i8qua_template, i8qua_answers, i8qua_kernel_fill_mode, i8qua_num_inputs, i8qua_num_output, i8qua_num_batch);
    }
    if (mode & 0x0010)/* Int4InputsInt8Weights */{
        const int i4i8_template[] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
        int i4i8_kernel_fill_mode = 2;
        const int8_t i4i8_answers[] = {
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79
        };
        const int   i4i8_num_inputs    = 32,
                    i4i8_num_output    = 32,
                    i4i8_num_batch     = 4;
        run_i4i8_tests(i4i8_template, i4i8_answers, i4i8_kernel_fill_mode, i4i8_num_inputs, i4i8_num_output, i4i8_num_batch);
    }
    if (mode & 0x0020)/* Int4InputsInt4Weights */{
        const int i4i4_template[] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
        int i4i4_kernel_fill_mode = 2;
        const int8_t i4i4_answers[] = {
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79, 
            11, 28, 45, 62, 79, 80, -79, -62, -45, -28, -11, 11, 28, 45, 62, 79
        };
        const int   i4i4_num_inputs    = 32,
                    i4i4_num_output    = 32,
                    i4i4_num_batch     = 4;
        run_i4i4_tests(i4i4_template, i4i4_answers, i4i4_kernel_fill_mode, i4i4_num_inputs, i4i4_num_output, i4i4_num_batch);
    }
    if (mode & 0x0080)/* BinaryInputsInt8Weights */{
        const int bini8_template[] = {
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,  
        };
        int bini8_kernel_fill_mode = 0;
        const int8_t bini8_answers[] = {
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0
        };
        const int   bini8_num_inputs    = 128,
                    bini8_num_output    = 128,
                    bini8_num_batch     = 4;
        run_bini8_tests(bini8_template, bini8_answers, bini8_kernel_fill_mode, bini8_num_inputs, bini8_num_output, bini8_num_batch);
    }
    if (mode & 0x0100)/* BinaryInputsBinaryWeights */{
        const int binbin_template[] = {
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 
            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,  
        };
        int binbin_kernel_fill_mode = 0;
        const int8_t binbin_answers[] = {
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0
        };
        const int   binbin_num_inputs    = 128,
                    binbin_num_output    = 128,
                    binbin_num_batch     = 4;
        run_binbin_tests(binbin_template, binbin_answers, binbin_kernel_fill_mode, binbin_num_inputs, binbin_num_output, binbin_num_batch);
    }
    if (mode & 0x0040)/* TernaryInputsInt8Weights */{
        const int teri8_template[] = {
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 
        };
        int teri8_kernel_fill_mode = 2;
        const int8_t teri8_answers[] = {
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
        };
        const int   teri8_num_inputs    = 64,
                    teri8_num_output    = 64,
                    teri8_num_batch     = 4;
        run_teri8_tests(teri8_template, teri8_answers, teri8_kernel_fill_mode, teri8_num_inputs, teri8_num_output, teri8_num_batch);
    }
    if (mode & 0x0200)/* TernaryInputsTernaryWeights */{
        const int terter_template[] = {
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 
            -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0,   
        };
        int terter_kernel_fill_mode = 0;
        const int8_t terter_answers[] = {
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
            -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, -1, 85, 0, 0,
        };
        const int   terter_num_inputs    = 64,
                    terter_num_output    = 64,
                    terter_num_batch     = 4;
        run_terter_tests(terter_template, terter_answers, terter_kernel_fill_mode, terter_num_inputs, terter_num_output, terter_num_batch);
    }
    if (mode & 0x0400)/* Int3InputsInt3Weights */{
        const int i3i3_template[] = {
            -2, -1, 0, 1, 2, -1, 0, 1, -2, -1, 0, 1, 2, -1, 0, 1, 
            -2, -1, 0, 1, 2, -1, 0, 1, -2, -1, 0, 1, 2, -1, 0, 1,
            -2, -1, 0, 1, 2, -1, 0, 1, -2, -1, 0, 1, 2, -1, 0, 1,
            -2, -1, 0, 1, 2, -1, 0, 1, -2, -1, 0, 1, 2, -1, 0, 1,  
            -2, -1, 0, 1, 2, -1, 0, 1, -2, -1, 0, 1, 2, -1, 0, 1, 
            -2, -1, 0, 1, 2, -1, 0, 1, -2, -1, 0, 1, 2, -1, 0, 1,
            -2, -1, 0, 1, 2, -1, 0, 1, -2, -1, 0, 1, 2, -1, 0, 1,
            -2, -1, 0, 1, 2, -1, 0, 1, -2, -1, 0, 1, 2, -1, 0, 1,  
        };
        int i3i3_kernel_fill_mode = 0;
        const int8_t i3i3_answers[] = {
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
            108, -37, -2, -1, 0, 0, -110, 36, 36, 73, -2, -1, 0, 0, -110, 36,
        };
        const int   i3i3_num_inputs    = 40,
                    i3i3_num_output    = 40,
                    i3i3_num_batch     = 4;
        run_i3i3_tests(i3i3_template, i3i3_answers, i3i3_kernel_fill_mode, i3i3_num_inputs, i3i3_num_output, i3i3_num_batch);
    }

    if (test_mul_api){
        if (test_mul_api & 0x0001)
            run_mul_api_tests(LowPrecision::Method::kInt8Int4);
        if (test_mul_api & 0x0002)
            run_mul_api_tests(LowPrecision::Method::kInt8Binary);
        if (test_mul_api & 0x0004)
            run_mul_api_tests(LowPrecision::Method::kInt8Ternary);
        if (test_mul_api & 0x0008)
            run_mul_api_tests(LowPrecision::Method::kInt8QuaTernary);
        if (test_mul_api & 0x0010)
            run_mul_api_tests(LowPrecision::Method::kInt4ActInt8Weight);
        if (test_mul_api & 0x0020)
            run_mul_api_tests(LowPrecision::Method::kInt4ActInt4Weight);
        if (test_mul_api & 0x0040)
            run_mul_api_tests(LowPrecision::Method::kTernaryActInt8Weight);
        if (test_mul_api & 0x0200)
            run_mul_api_tests(LowPrecision::Method::kTernaryActTernaryWeight);
        if (test_mul_api & 0x0080)
            run_mul_api_tests(LowPrecision::Method::kBinaryActInt8Weight);
        if (test_mul_api & 0x0100)
            run_mul_api_tests(LowPrecision::Method::kBinaryActBinaryWeight);
        if (test_mul_api & 0x0400)
            run_mul_api_tests(LowPrecision::Method::kInt3ActInt3Weight);
    }

    benchmark_mode_t benchmark_mode;

    benchmark_mode.multibatch_benchmark                                 = multibatch_benchmark_enable;
    benchmark_mode.singlebatch_benchmark                                = singlebatch_benchmark_enable;
    benchmark_mode.selected_benchmark_mode                              = selected_benchmark_enable;

    benchmark_mode.real_mul_api_benchmark_enable                        = selected_benchmark_real_mul_api != 0;
    benchmark_mode.real_mul_api_benchmark_mode                          = selected_benchmark_real_mul_api;

    benchmark_mode.real_single_mul_api_benchmark_enable                 = selected_benchmark_real_single_mul_api != 0;
    benchmark_mode.real_single_mul_api_benchmark_mode                   = selected_benchmark_real_single_mul_api;

    benchmark_mode.real_multi_mul_api_benchmark_enable                  = selected_benchmark_real_multi_mul_api != 0;
    benchmark_mode.real_multi_mul_api_benchmark_mode                    = selected_benchmark_real_multi_mul_api;

    benchmark_mode.single_mul_api_increasing_size_benchmark_enable      = enable_single_mul_api_increasing_size_benchmark != 0;
    benchmark_mode.single_mul_api_increasing_size_benchmark_mode        = enable_single_mul_api_increasing_size_benchmark;

    benchmark_mode.single_mul_api_different_size_benchmark_enable       = enable_single_mul_api_different_size_benchmark != 0;
    benchmark_mode.single_mul_api_different_size_benchmark_mode         = enable_single_mul_api_different_size_benchmark;
    benchmark_mode.single_mul_api_different_size_benchmark_time_path    = single_mul_api_different_size_benchmark_time_file;
    benchmark_mode.single_mul_api_different_size_benchmark_speedup_path = single_mul_api_different_size_benchmark_speedup_file;

    benchmark_mode.multi_mul_api_different_size_benchmark_enable        = enable_multi_mul_api_different_size_benchmark != 0;
    benchmark_mode.multi_mul_api_different_size_benchmark_mode          = enable_multi_mul_api_different_size_benchmark;
    benchmark_mode.multi_mul_api_different_size_benchmark_time_path     = multi_mul_api_different_size_benchmark_time_file;
    benchmark_mode.multi_mul_api_different_size_benchmark_speedup_path  = multi_mul_api_different_size_benchmark_speedup_file;

    run_benchmark(benchmark_iterations, benchmark_mode);

    return 0;
}
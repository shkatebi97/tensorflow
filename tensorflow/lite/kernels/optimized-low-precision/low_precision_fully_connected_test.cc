#include "low_precision_fully_connected.h"
#include "ruy/ruy.h"
#include "ruy/context.h"
#include <stdio.h>
#include <string>

using namespace std;
using namespace LowPrecision;
using namespace LowPrecision::FullyConnected;

#define DEBUG false
#define PRINT_ACTIVATIONS false
#define PRINT_MUL_OUTPUT true
#define PRINT_MUL_MB_OUTPUT false
#define PRINT_VALUES_IN_HEX false
#define PRINT_VALUES false

void run_i8i4_tests(
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
        _filter_shape[2]      = { num_output, num_inputs / 2 },
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
    Status ret = LowPrecision::FullyConnected::Int4::QuantizeFilter(kernel_data, kernel_shape, filter_data, MemLayout::kRowMajor);
    cout << "LowPrecision::FullyConnected::Int4::QuantizeFilter-RowMajor Return Status \t=> " << ((ret)?("\033[1m\033[31m"):("\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 512){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4RMPassed &= (_answers[i] == filter_data[i]);
        cout << "LowPrecision::FullyConnected::Int4::QuantizeFilter-RowMajor    \t\t\t=> " << ((QFTI4RMPassed)?("\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
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
    ret = LowPrecision::FullyConnected::Int4::QuantizeFilter(kernel_data_C, kernel_shape, filter_data, MemLayout::kColumnMajor);
    cout << "LowPrecision::FullyConnected::Int4::QuantizeFilter-ColumnMajor Return Status \t=> " << ((ret)?("\033[1m\033[31m"):("\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 512){
        bool QFTI4CMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4CMPassed &= (_answers[i] == filter_data[i]);
        cout << "LowPrecision::FullyConnected::Int4::QuantizeFilter-ColumnMajor \t\t\t=> " << ((QFTI4CMPassed)?("\033[32m"):("\033[1m\033[31m")) << (QFTI4CMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
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

    minus_one_vector(output_data, output_shape.flatsize);
    ret = LowPrecision::FullyConnected::Int4::MultiplyInt8(input_data, input_shape,
                            filter_data, kernel_shape,
                            output_data, output_shape);
    bool MI8I4Passed = true;
    for (int i = 0 ; i < output_shape.flatsize / 2 ; i++)
        MI8I4Passed &= (output_data_R[i] == output_data[i]);

    cout << "LowPrecision::FullyConnected::Int4::MultiplyInt8 Return Status \t\t\t=> " << ((ret)?("\033[1m\033[31m"):("\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    cout << "LowPrecision::FullyConnected::Int4::MultiplyInt8 \t\t\t\t=> " << ((MI8I4Passed)?("\033[32m"):("\033[1m\033[31m")) << (MI8I4Passed?"PASSED":"FAILED") << "\033[0m" << endl;
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
    ret = LowPrecision::FullyConnected::Int4::MultiplyInt8(input_pack_MB, input_shape_MB,
                            filter_data, kernel_shape,
                            output_data_MB, output_shape_MB);
    bool MI8I4MBPassed = true;
    for (int i = 0 ; i < output_shape_MB.flatsize / 2 ; i++)
        MI8I4MBPassed &= (output_data_R_MB[i] == output_data_MB[i]);
    cout << "LowPrecision::FullyConnected::Int4::MultiplyInt8-MB Return Status \t\t=> " << ((ret)?("\033[1m\033[31m"):("\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    cout << "LowPrecision::FullyConnected::Int4::MultiplyInt8-MB \t\t\t\t=> " << ((MI8I4MBPassed)?("\033[32m"):("\033[1m\033[31m")) << (MI8I4MBPassed?"PASSED":"FAILED") << "\033[0m" << endl;
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
    cout << "LowPrecision::FullyConnected::Int4  Deallocation\t\t\t\t=> \033[32mPASSED\033[0m" << endl;
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
    cout << "LowPrecision::FullyConnected::Binary::QuantizeFilter-RowMajor Return Status \t=> " << ((ret)?("\033[1m\033[31m"):("\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 128 * 16){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4RMPassed &= (_answers[i] == filter_data[i]);
        cout << "LowPrecision::FullyConnected::Binary::QuantizeFilter-RowMajor    \t\t=> " << ((QFTI4RMPassed)?("\033[32m"):("\033[1m\033[31m")) << (QFTI4RMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
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
    cout << "LowPrecision::FullyConnected::Binary::QuantizeFilter-ColumnMajor Return Status \t=> " << ((ret)?("\033[1m\033[31m"):("\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    if (filter_shape.flatsize <= 128 * 16){
        bool QFTI4CMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4CMPassed &= (_answers[i] == filter_data[i]);
        cout << "LowPrecision::FullyConnected::Binary::QuantizeFilter-ColumnMajor \t\t=> " << ((QFTI4CMPassed)?("\033[32m"):("\033[1m\033[31m")) << (QFTI4CMPassed?"PASSED":"FAILED") << "\033[0m" << endl;
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

    cout << "LowPrecision::FullyConnected::Binary::MultiplyInt8 Return Status \t\t=> " << ((ret)?("\033[1m\033[31m"):("\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
    cout << "LowPrecision::FullyConnected::Binary::MultiplyInt8 \t\t\t\t=> " << ((MI8I4Passed)?("\033[32m"):("\033[1m\033[31m")) << (MI8I4Passed?"PASSED":"FAILED") << "\033[0m" << endl;
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
//     ret = LowPrecision::FullyConnected::Binary::MultiplyInt8(input_pack_MB, input_shape_MB,
//                             filter_data, kernel_shape,
//                             output_data_MB, output_shape_MB);
//     bool MI8I4MBPassed = true;
//     for (int i = 0 ; i < output_shape_MB.flatsize / 2 ; i++)
//         MI8I4MBPassed &= (output_data_R_MB[i] == output_data_MB[i]);
//     cout << "LowPrecision::FullyConnected::Binary::MultiplyInt8-MB Return Status \t\t=> " << ((ret)?("\033[1m\033[31m"):("\033[32m")) << ((ret)?("FAILED"):("PASSED")) << "\033[0m" << endl;
//     cout << "LowPrecision::FullyConnected::Binary::MultiplyInt8-MB \t\t\t\t=> " << ((MI8I4MBPassed)?("\033[32m"):("\033[1m\033[31m")) << (MI8I4MBPassed?"PASSED":"FAILED") << "\033[0m" << endl;
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
    cout << "LowPrecision::FullyConnected::Binary  Deallocation\t\t\t\t=> \033[32mPASSED\033[0m" << endl;
}

int main(){
    int mode = 0x02;
    if (mode & 0x01){
        const int i8i4_template[] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
        int i8i4_kernel_fill_mode = 0;
        const int8_t i8i4_answers[] = {
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
        const int   i8i4_num_inputs    = 32,
                    i8i4_num_output    = 32,
                    i8i4_num_batch     = 4;
        run_i8i4_tests(i8i4_template, i8i4_answers, i8i4_kernel_fill_mode, i8i4_num_inputs, i8i4_num_output, i8i4_num_batch);
    }
    if (mode & 0x02){
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
    return 0;
}
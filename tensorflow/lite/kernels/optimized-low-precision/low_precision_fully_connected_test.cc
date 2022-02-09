#include "low_precision_fully_connected.h"
#include "ruy/ruy.h"
#include "ruy/context.h"
#include <stdio.h>
#include <string>

using namespace std;
using namespace LowPrecision;
using namespace LowPrecision::FullyConnected;

#define DEBUG false
#define PRINT_MUL_OUTPUT false
#define PRINT_VALUES_IN_HEX false
#define PRINT_VALUES false

int main(){
    const int _template[] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
    int kernel_fill_mode = 0;
    const int8_t _answers[] = {
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
    int num_inputs = 2048 * 2,
        num_output = 2048 * 4;
    int _input_shape[1]  = { num_inputs },
        _kernel_shape[2] = { num_output, num_inputs },
        _filter_shape[2] = { num_output, num_inputs / 2 },
        _output_shape[1] = { num_output };
    Shape input_shape  = get_shape(_input_shape,  1),
          kernel_shape = get_shape(_kernel_shape, 2),
          filter_shape = get_shape(_filter_shape, 2),
          output_shape = get_shape(_output_shape, 1);
    
    int8_t*   input_data   = allocate<int8_t>(input_shape.flatsize);
    int8_t*  kernel_data   = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  kernel_data_C = allocate<int8_t>(kernel_shape.flatsize);
    int8_t*  filter_data   = allocate<int8_t>(filter_shape.flatsize);
    int32_t* output_data   = allocate<int32_t>(output_shape.flatsize);
    int32_t* output_data_R = allocate<int32_t>(output_shape.flatsize);

    for (int i = 0; i < num_inputs; i++)
        input_data[i] = 1;
    
    if(kernel_fill_mode == 0){
        for (int i = 0; i < kernel_shape.size[0]; i++)
            for (int j = 0; j < kernel_shape.size[1]; j++)
                kernel_data[i * kernel_shape.size[1] + j] = _template[j % 32];

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
    Status ret = QuantizeFilterToInt4(kernel_data, kernel_shape, filter_data, MemLayout::kRowMajor);
    cout << "QuantizeFilterToInt4-RowMajor Return Status \t=> " << ((ret)?("FAILED"):("PASSED")) << endl;
    if (filter_shape.flatsize <= 512){
        bool QFTI4RMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4RMPassed &= (_answers[i] == filter_data[i]);
        cout << "QuantizeFilterToInt4-RowMajor    \t\t=> " << (QFTI4RMPassed?"PASSED":"FAILED") << endl;
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
    ret = QuantizeFilterToInt4(kernel_data_C, kernel_shape, filter_data, MemLayout::kColumnMajor);
    cout << "QuantizeFilterToInt4-ColumnMajor Return Status \t=> " << ((ret)?("FAILED"):("PASSED")) << endl;
    if (filter_shape.flatsize <= 512){
        bool QFTI4CMPassed = true;
        for (int i = 0 ; i < filter_shape.flatsize ; i++)
            QFTI4CMPassed &= (_answers[i] == filter_data[i]);
        cout << "QuantizeFilterToInt4-ColumnMajor \t\t=> " << (QFTI4CMPassed?"PASSED":"FAILED") << endl;
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
    ruy::MakeSimpleLayout(kernel_shape.size[1], kernel_shape.size[1], ruy::Order::kColMajor,
                              ruy_rhs.mutable_layout());
    ruy_rhs.set_data(kernel_data);
    ruy_rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    // Create rhs
    ruy::MakeSimpleLayout(kernel_shape.size[1], kernel_shape.size[1], ruy::Order::kRowMajor,
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
    ret = MultiplyInt8Int4(input_data, input_shape,
                            filter_data, kernel_shape,
                            output_data, output_shape);
    bool MI8I4Passed = true;
    for (int i = 0 ; i < output_shape.flatsize / 2 ; i++)
        MI8I4Passed &= (output_data_R[i] == output_data[i]);

    cout << "MultiplyInt8Int4 Return Status \t\t\t=> " << ((ret)?("FAILED"):("PASSED")) << endl;
    cout << "MultiplyInt8Int4 \t\t\t\t=> " << (MI8I4Passed?"PASSED":"FAILED") << endl;
#if PRINT_VALUES || PRINT_MUL_OUTPUT
    cout << "[";
    for (int i = 0; i < output_shape.size[0]; i++)
        cout << ((int)output_data[i]) << ", ";
    cout << "]";
    cout << endl;
#endif

    return 0;
}
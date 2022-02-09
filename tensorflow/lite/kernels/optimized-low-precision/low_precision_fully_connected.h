#ifndef LOW_PRECISION_FULLY_CONNECTED_H_
#include "common/types.h"
#include "ops-implementations/mul/LowPrecisionPacking.h"
#include <string>
#include <iostream>
#include <vector>
#include <tuple>

#ifdef IS_ARM
#include <arm_neon.h>
#endif

#ifdef IS_ARM
namespace LowPrecision{
    namespace FullyConnected{
        LowPrecision::Method GetMethodFromEnv();
        std::string GetVariableFromEnv(std::string variable);
        LowPrecision::DataType GetDataType(int type);
        bool IsAppliable(
            LowPrecision::Method method, LowPrecision::Shape input_shape, 
            LowPrecision::DataType input_type, LowPrecision::DataType filter_type, LowPrecision::DataType output_type);
        LowPrecision::Status QuantizeFilterToInt4(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
        LowPrecision::Status MultiplyInt8Int4(
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape);
        void do4BitMultiplication(const int8_t* activation, 
                                int8_t* weights, 
                                int32_t* dst_1, int size);
        uint8_t quantizeTo4BitIntAndPackBitsStep(const int8_t& input, int shift_amount);
    }
}
#else
namespace LowPrecision{
    namespace FullyConnected{
        bool IsAppliable(
            LowPrecision::Method method, LowPrecision::Shape input_shape, 
            LowPrecision::DataType input_type, LowPrecision::DataType filter_type, LowPrecision::DataType output_type) { return false; }
        LowPrecision::Method GetMethodFromEnv() { return Method::kNoOptimization; }
        std::string GetVariableFromEnv(std::string variable) { return std::string(); }
        LowPrecision::DataType GetDataType(int type) { return LowPrecision::DataType::NotAvailable; }
        LowPrecision::Status QuantizeFilterToInt4(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout) { return LowPrecision::Status::NotSupported; }
        LowPrecision::Status MultiplyInt8Int4(
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape) { return LowPrecision::Status::NotSupported; }
        void do4BitMultiplication(const int8_t* activation, 
                                int8_t* weights, 
                                int32_t& dst_1, int32_t& dst_2,
                                int32_t& dst_3, int32_t& dst_4, 
                                int size) {}
        uint8_t quantizeTo4BitIntAndPackBitsStep(const int8_t& input, int shift_amount){ return 0; }
    }
}
#endif
#endif
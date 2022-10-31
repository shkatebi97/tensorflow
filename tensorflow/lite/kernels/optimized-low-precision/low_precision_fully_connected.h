#ifndef LOW_PRECISION_FULLY_CONNECTED_H_
#define LOW_PRECISION_FULLY_CONNECTED_H_
#include "common/types.h"
#include "ops-implementations/mul/LowPrecisionPacking.h"
#include "common/flags.h"
#include <string>
#include <iostream>
#include <vector>
#include <tuple>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>

#ifdef IS_ARM
#include <arm_neon.h>
#endif

#ifdef IS_ARM
// #define PRINT_VALUES true
// #define PRINT_VALUES_DETAILED false
namespace LowPrecision {
    class Matrix {
        int8_t*                  _data                   = nullptr;
        int8_t*                  _scratchpad             = nullptr;
        bool                     _data_is_in_scratchpad  = false;
        bool                     _need_scratchpad        = false;
        LowPrecision::MemLayout  _mem_layout             = LowPrecision::MemLayout::kColumnMajor;
        LowPrecision::Shape      _shape;
        LowPrecision::MatrixType _type;
        bool                     _need_downcast          = false;
        int32_t                  _downcast_coeff         = 1;
    public:
        Matrix(
            LowPrecision::MatrixType type = LowPrecision::MatrixType::Unknown,
            LowPrecision::MemLayout memLayout = LowPrecision::MemLayout::kColumnMajor
        ){ _mem_layout = memLayout; _type = type; }
        Matrix(const Matrix& var) {
            this->_data                     = var._data;
            this->_mem_layout               = var._mem_layout;
            this->_shape                    = var._shape;
            this->_scratchpad               = var._scratchpad;
            this->_data_is_in_scratchpad    = var._data_is_in_scratchpad;
            this->_need_scratchpad          = var._need_scratchpad;
            this->_type                     = var._type;
        }
    
        bool isScratchpadValid()                                        { return _data_is_in_scratchpad; }
        bool getNeedScratchpad()                                        { return _need_scratchpad; }
        bool getNeedDowncast()                                          { return _need_downcast; }
        int8_t* getData()                                               { return _data; }
        int8_t* getScratchpad()                                         { return _scratchpad; }
        int32_t getDowncastCoeff()                                      { return _downcast_coeff; }
        LowPrecision::Shape getShape()                                  { return _shape; }
        LowPrecision::MemLayout getMemLayout()                          { return _mem_layout; }
        LowPrecision::MatrixType getMatrixType()                        { return _type; }
    
        void setScratchpadValid(bool enable_scratchpad = true)          { _data_is_in_scratchpad= enable_scratchpad; }
        void setNeedScratchpad(bool need_scratchpad = true)             { _need_scratchpad      = need_scratchpad; }
        void setData(int8_t* data)                                      { _data                 = data; }
        void setScratchpad(int8_t* data)                                { _scratchpad           = data; }
        void setData(int32_t* data)                                     { _data                 = LowPrecision::get_pointer_as<int8_t>(data); }
        void setScratchpad(int32_t* data)                               { _scratchpad           = LowPrecision::get_pointer_as<int8_t>(data); }
        void setDataAndScratchpad(int8_t* data, int8_t* scratchpad)     { _data                 = data;
                                                                          _scratchpad           = scratchpad; }
        void setDataAndScratchpad(int32_t* data, int32_t* scratchpad)   { _data                 = LowPrecision::get_pointer_as<int8_t>(data);
                                                                          _scratchpad           = LowPrecision::get_pointer_as<int8_t>(scratchpad); }
        void setDataAndScratchpadAndShape(const int8_t* data, const int8_t* scratchpad, LowPrecision::Shape shape)
                                                                        { _data                 = const_cast<int8_t*>(data);
                                                                          _scratchpad           = const_cast<int8_t*>(scratchpad);
                                                                          _shape                = shape; }
        void setDataAndScratchpadAndShape(const int32_t* data, const int32_t* scratchpad, LowPrecision::Shape shape)
                                                                        { _data                 = LowPrecision::get_pointer_as<int8_t>(const_cast<int32_t*>(data));
                                                                          _scratchpad           = LowPrecision::get_pointer_as<int8_t>(const_cast<int32_t*>(scratchpad));
                                                                          _shape                = shape; }
        void setShape(LowPrecision::Shape shape)                        { _shape                = shape; }
        void setMemLayout(LowPrecision::MemLayout mem_layout)           { _mem_layout           = mem_layout; }
        void setDowncastCoeff(int32_t downcast_coeff)                   { _downcast_coeff = downcast_coeff; 
                                                                          _need_downcast = true; }
    };
    class Params{
    public:
        int start_batches;
        int start_columns;
        int start_rows;

        int end_batches;
        int end_columns;
        int end_rows;

        int lhs_stride;
        int rhs_stride;
        int dst_stride;
        Params(){}
    };
    namespace FullyConnected {
        static long int id = 0;
        LowPrecision::Method get_default_method();
        void set_default_method(LowPrecision::Method method);
        LowPrecision::Method GetMethodFromEnv();
        std::string GetVariableFromEnv(std::string variable);
        LowPrecision::DataType GetDataType(int type);
        bool IsAppliable(
            LowPrecision::Method method, LowPrecision::Shape input_shape, LowPrecision::Shape filter_shape, 
            LowPrecision::DataType input_type, LowPrecision::DataType filter_type,
            LowPrecision::DataType output_type, bool Is_FC);
        bool IncludesActivationCompression(LowPrecision::Method method);
        size_t CalcFlatSize(int* sizes, int num_dims);
        int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape, Method method);
        size_t TransformFilterShape(LowPrecision::Method method, int* shape, int n_dims);
        size_t TransformInputShape(LowPrecision::Method method, int* shape, int n_dims);
        LowPrecision::Status QuantizeFilter(LowPrecision::Method method, const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
        LowPrecision::Status QuantizeInput(LowPrecision::Method method, const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
        LowPrecision::Status Multiply(
            LowPrecision::Method method,
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape);
        LowPrecision::Status MultiplyInt8SingleBatch(
            LowPrecision::Method method,
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape);
        LowPrecision::Status MultiplyInt8MultiBatched(
            LowPrecision::Method method,
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape);
        LowPrecision::Status MultiplyInt8MultiBatchedBlockProcessing(
            LowPrecision::Method method,
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape);
        Shape GetPaddedShape(const LowPrecision::Method method, const Shape& input_shape, bool pad_rows_too = false);
        Status PadMatrixFromShapeToShape(const int8_t* input, int8_t* output, Shape from_shape, Shape to_shape, const int8_t pad_value = 0);
        template<typename T>
        Status DePadMatrixFromShapeToShape(const T* input, T* output, Shape from_shape, Shape to_shape);
        Status ApplyDowncast(int32_t* input, int8_t* output, Shape shape, const int32_t downcast_coeff);

        namespace Int4 {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            Status PaddingWeightsIfNeeded(const int8_t* input_weight, int8_t* output_weight, Shape shape);
            Status PaddingInputsIfNeeded(const int8_t* input, int8_t* output, Shape shape);
            Shape GetPaddedShape(const Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilterWithPadding(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInputWithPadding(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status MultiplyInt8(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params);
            void doMultiplication1Col(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int size);
            void doMultiplication(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int32_t* dst_2,
                                    int32_t* dst_3, int32_t* dst_4,
                                    int size);
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        namespace Binary {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params);
            void doMultiplication1Col(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int size);
            void doMultiplication(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int32_t* dst_2,
                                    int32_t* dst_3, int32_t* dst_4,
                                    int size);
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        namespace Ternary {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params);
            void doMultiplication1Col(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int size);
            void doMultiplication(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int32_t* dst_2,
                                    int32_t* dst_3, int32_t* dst_4,
                                    int size);
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        namespace Quaternary {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params);
            void doMultiplication1Col(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int size);
            void doMultiplication(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int32_t* dst_2,
                                    int32_t* dst_3, int32_t* dst_4,
                                    int size);
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        namespace Int4InputsInt8Weights {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params);
            void doMultiplication1Col(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int size);
            void doMultiplication(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int32_t* dst_2,
                                    int32_t* dst_3, int32_t* dst_4,
                                    int size);
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        namespace Int4InputsInt4Weights {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params);
            void doMultiplication1Col(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int size);
            void doMultiplication(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int32_t* dst_2,
                                    int32_t* dst_3, int32_t* dst_4,
                                    int size);
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        namespace BinaryInputsInt8Weights {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        namespace BinaryInputsBinaryWeights {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        namespace BinaryInputsBinaryWeightsXOR {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        namespace TernaryInputsInt8Weights {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params);
            void doMultiplication1Col(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int size);
            void doMultiplication(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int32_t* dst_2,
                                    int32_t* dst_3, int32_t* dst_4,
                                    int size);
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        namespace TernaryInputsTernaryWeights {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params);
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        namespace Int3InputsInt3Weights {
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        namespace Int8InputsInt4PowerWeights {
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape);
        }
        namespace ULPPACK {
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                size_t Wb, size_t Ab
            );
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape,
                size_t Wb, size_t Ab
            );
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params);
        }
        
        
        void doScallingFactorMultiplication(int32_t* input, const float* scalling_factor, float* output,
                                            int batch_n, int input_n);
        LowPrecision::Status Mul(Matrix& lhs, Matrix& rhs, Matrix& dst, Method method);
    }
}
#else
namespace LowPrecision{
    namespace FullyConnected{
        bool IsAppliable(
            LowPrecision::Method method, LowPrecision::Shape input_shape, 
            LowPrecision::DataType input_type, LowPrecision::DataType filter_type,
            LowPrecision::DataType output_type, bool Is_FC) { return false; }
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
                                int size) {  }
        uint8_t quantizeTo4BitIntAndPackBitsStep(const int8_t& input, int shift_amount){ return 0; }
        void doScallingFactorMultiplication(int32_t* input, const float* scalling_factor, float* output,
                                            int batch_n, int input_n){ }
    }
}
#endif
#endif
#include "low_precision_fully_connected.h"

#ifndef IS_ARM
#else
namespace LowPrecision{
    long int Shape::last_id = 0;
    
    namespace FullyConnected{
        using ::LowPrecision::Method;
        using ::LowPrecision::Shape;
        using ::LowPrecision::Status;
        using ::LowPrecision::DataType;
        using ::LowPrecision::MemLayout;
        using ::LowPrecision::Matrix;
        using ::LowPrecision::Params;
        using ::LowPrecision::MatrixType;
        LowPrecision::Method __default_method;
        LowPrecision::Method get_default_method() { return __default_method; } 
        void set_default_method(LowPrecision::Method method) { __default_method = method; }
        Method GetMethodFromEnv(){
            char * val;
            val = getenv( "LowPrecisionFC" );
            std::string retval = "";                                                          
            if (val != NULL) {
                retval = val;
            }
            if (retval == std::string("I8-I4"))
                return Method::kInt8Int4;
            else if (retval == std::string("I8-Binary"))
                return Method::kInt8Binary;
            else if (retval == std::string("I8-Ternary"))
                return Method::kInt8Ternary;
            else if (retval == std::string("I8-Quaternary"))
                return Method::kInt8QuaTernary;
            else if (retval == std::string("F32-Binary"))
                return Method::kFloat32Binary;
            else if (retval == std::string("I4-I8"))
                return Method::kInt4ActInt8Weight;
            else if (retval == std::string("I4-I4"))
                return Method::kInt4ActInt4Weight;
            else if (retval == std::string("Ternary-I8"))
                return Method::kTernaryActInt8Weight;
            else if (retval == std::string("Ternary-Ternary"))
                return Method::kTernaryActTernaryWeight;
            else if (retval == std::string("Binary-I8"))
                return Method::kBinaryActInt8Weight;
            else if (retval == std::string("Binary-Binary"))
                return Method::kBinaryActBinaryWeight;
            else if (retval == std::string("Binary-Binary-XOR"))
                return Method::kBinaryActBinaryWeightXOR;
            else if (retval == std::string("I3-I3"))
                return Method::kInt3ActInt3Weight;
            else if (retval == std::string("ULPPACK-W1A1"))
                return Method::kULPPACKW1A1;
            else if (retval == std::string("ULPPACK-W2A2"))
                return Method::kULPPACKW2A2;
            else if (retval == std::string("ULPPACK-W3A3"))
                return Method::kULPPACKW3A3;
            else if (retval == std::string("ULPPACK-W4A4"))
                return Method::kULPPACKW4A4;
            else if (retval == std::string("ULPPACK-W5A5"))
                return Method::kULPPACKW5A5;
            else if (retval == std::string("ULPPACK-W6A6"))
                return Method::kULPPACKW6A6;
            else if (retval == std::string("ULPPACK-W7A7"))
                return Method::kULPPACKW7A7;
            else
                return Method::kNoOptimization;
        } 
        std::string GetVariableFromEnv(std::string variable){
            char * val;
            val = getenv( variable.c_str() );
            std::string retval = "";                                                          
            if (val != NULL) {
                retval = val;
            }
            return retval;
        }
        LowPrecision::DataType GetDataType(int type){
            if (type == 1)
                return DataType::Float32;
            else if (type == 2)
                return DataType::Int32;
            else if (type == 6)
                return DataType::Bool;
            else if (type == 7)
                return DataType::Int16;
            else if (type == 9)
                return DataType::Int8;
            else if (type == 10)
                return DataType::Float16;
            else
                return DataType::NotAvailable;
        }
        bool IsAppliable(
            Method method, Shape input_shape, Shape filter_shape,
            DataType input_type, DataType filter_type, 
            DataType output_type, bool Is_FC = false){
            bool multibatched_enabled = !(GetVariableFromEnv( "LowPrecisionMultiBatched" ) == "FALSE") || method & Method::kULPPACK;
            bool singlebatched_enabled = !(GetVariableFromEnv( "LowPrecisionSingleBatched" ) == "FALSE");
            bool input_row_pad_enabled = !(GetVariableFromEnv( "LowPrecisionPadInputRows" ) == "FALSE");
            bool is_multibatched = input_shape.number_dims == 2 && input_shape.size[0] > 1;
            // Checking for Not-Supported Input DataTypes
            if (
                (input_type != DataType::Int8 && input_type != DataType::Float32 && input_type != DataType::Int32) ||
                filter_type != DataType::Int8 ||
                (output_type != DataType::Int8 && output_type != DataType::Float32 && output_type != DataType::Int32))
                return false;
            // Checking for the conditions of rejection of multi-batched or single-batch input
            if(is_multibatched && !multibatched_enabled)
                return false;
            else if(!is_multibatched && !singlebatched_enabled)
                return false;
            
            if (
                is_multibatched && 
                multibatched_enabled &&
                input_shape.size[0] % 4 &&
                !input_row_pad_enabled
            )
                return false;

            // Checking common conditions
            if (filter_shape.size[filter_shape.number_dims - 2] % 4)
                return false;

            // Checking conditions of input shape of any method
            // if (method == Method::kInt8Binary ||
            //     method == Method::kFloat32Binary ||
            //     method == Method::kFloat16Binary)
            //     return true || !(input_shape.size[input_shape.number_dims - 1] % 128);
            // if (method == Method::kInt8Ternary ||
            //     method == Method::kFloat32Ternary ||
            //     method == Method::kFloat16Ternary ||
            //     method == Method::kInt8QuaTernary)
            //     return true || !(input_shape.size[input_shape.number_dims - 1] % 64);
            // if (method == Method::kInt8Int4)
            //     return true || !(input_shape.size[input_shape.number_dims - 1] % 32);
            // if (method == Method::kInt4ActInt8Weight)
            //     return true || !(input_shape.size[input_shape.number_dims - 1] % 32);
            // if (method == Method::kInt4ActInt4Weight)
            //     return true || !(input_shape.size[input_shape.number_dims - 1] % 32);
            // if (method == Method::kTernaryActInt8Weight)
            //     return true || !(input_shape.size[input_shape.number_dims - 1] % 64);
            // if (method == Method::kTernaryActTernaryWeight)
            //     return true || !(input_shape.size[input_shape.number_dims - 1] % 64);
            // if (method == Method::kBinaryActInt8Weight)
            //     return true || !(input_shape.size[input_shape.number_dims - 1] % 128);
            // if (method == Method::kBinaryActBinaryWeight)
            //     return true || !(input_shape.size[input_shape.number_dims - 1] % 128);
            // if (method == Method::kBinaryActBinaryWeightXOR)
            //     return true || !(input_shape.size[input_shape.number_dims - 1] % 128);
            if (method == Method::kInt3ActInt3Weight)
                return !(input_shape.size[input_shape.number_dims - 1] % 40);
            // if (method & Method::kULPPACK)
            //     return !(input_shape.size[input_shape.number_dims - 1] % 16);
            if (method == Method::kNoOptimization)
                return false;
            // If none of the aboves
            return true;
        }
        bool IncludesActivationCompression(Method method){
            return 
                (method & Method::kInt4ActInt8Weight)        || 
                (method & Method::kInt4ActInt4Weight)        ||
                (method & Method::kTernaryActInt8Weight)     ||
                (method & Method::kTernaryActTernaryWeight)  ||
                (method & Method::kBinaryActInt8Weight)      ||
                (method & Method::kBinaryActBinaryWeight)    ||
                (method & Method::kBinaryActBinaryWeightXOR) ||
                (method & Method::kInt3ActInt3Weight)        ||
                (method & Method::kULPPACK)
                ;
        }
        size_t CalcFlatSize(int* sizes, int num_dims){
            size_t size = 1;
            for (size_t i = 0; i < num_dims; i++)
                size *= sizes[i];
            return size;
        }
        int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape, Method method){
            switch (method)
            {
            case Method::kInt8Int4:
                return Int4::PaddingWeightsIfNeeded(weight, shape);

            case Method::kFloat32Binary:
            case Method::kInt8Binary:
                return Binary::PaddingWeightsIfNeeded(weight, shape);

            case Method::kInt8Ternary:
                return Ternary::PaddingWeightsIfNeeded(weight, shape);

            case Method::kInt8QuaTernary:
                return Quaternary::PaddingWeightsIfNeeded(weight, shape);

            case Method::kInt4ActInt8Weight:
                return Int4InputsInt8Weights::PaddingWeightsIfNeeded(weight, shape);

            case Method::kInt4ActInt4Weight:
                return Int4InputsInt4Weights::PaddingWeightsIfNeeded(weight, shape);

            case Method::kTernaryActInt8Weight:
                return TernaryInputsInt8Weights::PaddingWeightsIfNeeded(weight, shape);

            case Method::kBinaryActInt8Weight:
                return BinaryInputsInt8Weights::PaddingWeightsIfNeeded(weight, shape);

            default:
                return nullptr;
            }
        }
        size_t TransformFilterShape(LowPrecision::Method method, int* shape, int n_dims){
            int least_dim_size = 16, reduction_coeff = 1;
            if (method == LowPrecision::Method::kInt4ActInt8Weight){
                least_dim_size  = 32;
                reduction_coeff = 1;
            }
            else if (method == LowPrecision::Method::kInt4ActInt4Weight){
                least_dim_size  = 32;
                reduction_coeff = 2;
            }
            else if (method == LowPrecision::Method::kTernaryActInt8Weight){
                least_dim_size  = 64;
                reduction_coeff = 1;
            }
            else if (method == LowPrecision::Method::kTernaryActTernaryWeight){
                least_dim_size  = 64;
                reduction_coeff = 4;
            }
            else if (method == LowPrecision::Method::kBinaryActInt8Weight){
                least_dim_size  = 128;
                reduction_coeff = 1;
            }
            else if (method == LowPrecision::Method::kBinaryActBinaryWeight){
                least_dim_size  = 128;
                reduction_coeff = 8;
            }
            else if (method == LowPrecision::Method::kBinaryActBinaryWeightXOR){
                least_dim_size  = 128;
                reduction_coeff = 8;
            }
            else if (method == LowPrecision::Method::kInt8Int4){
                least_dim_size  = 32;
                reduction_coeff = 2;
            }
            else if (method == LowPrecision::Method::kInt8Binary){
                least_dim_size  = 128;
                reduction_coeff = 8;
            }
            else if (method == LowPrecision::Method::kInt8Ternary){
                least_dim_size  = 64;
                reduction_coeff = 4;
            }
            else if (method == LowPrecision::Method::kInt8QuaTernary){
                least_dim_size  = 64;
                reduction_coeff = 4;
            }
            
            shape[n_dims - 1] = (::ceil(shape[n_dims - 1] / ((float)least_dim_size)) * least_dim_size) / reduction_coeff;
            return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
        }
        size_t TransformInputShape(LowPrecision::Method method, int* shape, int n_dims){
            int least_dim_size = 16, reduction_coeff = 1;
            int least_row_size = 4;
            if (method == LowPrecision::Method::kInt4ActInt8Weight){
                least_dim_size  = 32;
                reduction_coeff = 2;
            }
            else if (method == LowPrecision::Method::kInt4ActInt4Weight){
                least_dim_size  = 32;
                reduction_coeff = 2;
            }
            else if (method == LowPrecision::Method::kTernaryActInt8Weight){
                least_dim_size  = 64;
                reduction_coeff = 4;
            }
            else if (method == LowPrecision::Method::kTernaryActTernaryWeight){
                least_dim_size  = 64;
                reduction_coeff = 4;
            }
            else if (method == LowPrecision::Method::kBinaryActInt8Weight){
                least_dim_size  = 128;
                reduction_coeff = 8;
            }
            else if (method == LowPrecision::Method::kBinaryActBinaryWeight){
                least_dim_size  = 128;
                reduction_coeff = 8;
            }
            else if (method == LowPrecision::Method::kBinaryActBinaryWeightXOR){
                least_dim_size  = 128;
                reduction_coeff = 8;
            }
            else if (method == LowPrecision::Method::kInt8Int4){
                least_dim_size  = 32;
                reduction_coeff = 1;
            }
            else if (method == LowPrecision::Method::kInt8Binary){
                least_dim_size  = 128;
                reduction_coeff = 1;
            }
            else if (method == LowPrecision::Method::kInt8Ternary){
                least_dim_size  = 64;
                reduction_coeff = 1;
            }
            else if (method == LowPrecision::Method::kInt8QuaTernary){
                least_dim_size  = 64;
                reduction_coeff = 1;
            }

            shape[n_dims - 1] = (::ceil(shape[n_dims - 1] / ((float)least_dim_size)) * least_dim_size) / reduction_coeff;
            bool is_multibatch = n_dims >= 2 && shape[n_dims - 2] > 1;
            if (is_multibatch && !(method & LowPrecision::Method::kULPPACK))
                shape[n_dims - 2] = ::ceil(shape[n_dims - 2] / ((float)least_row_size)) * least_row_size;
            else if(method & LowPrecision::Method::kULPPACK)
                shape[n_dims - 2] = ::ceil(shape[n_dims - 2] / 8.0) * 8;
            return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
        }
        Status QuantizeFilter(LowPrecision::Method method, const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout){
            int8_t* input_ptr = const_cast<int8_t*>(input);
            Shape input_padded_shape;
            input_padded_shape = GetPaddedShape(method, k_shape);
            bool need_padding = input_padded_shape != k_shape;
            if (need_padding){
                input_ptr = ::LowPrecision::allocate<int8_t>(input_padded_shape.flatsize);
                Status pad_ret = PadMatrixFromShapeToShape(input, input_ptr, k_shape, input_padded_shape);
                if (pad_ret != Status::Success) return pad_ret;
            }
            LowPrecision::Status ret;
            if (method == LowPrecision::Method::kInt8Int4)
                ret = LowPrecision::FullyConnected::Int4::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kInt8Ternary)
                ret = LowPrecision::FullyConnected::Ternary::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kInt8QuaTernary)
                ret = LowPrecision::FullyConnected::Quaternary::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (
                method == LowPrecision::Method::kInt8Binary ||
                method == LowPrecision::Method::kFloat32Binary ||
                method == LowPrecision::Method::kFloat16Binary
            )
                ret = LowPrecision::FullyConnected::Binary::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kInt4ActInt8Weight)
                ret = LowPrecision::FullyConnected::Int4InputsInt8Weights::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kInt4ActInt4Weight)
                ret = LowPrecision::FullyConnected::Int4InputsInt4Weights::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kTernaryActInt8Weight)
                ret = LowPrecision::FullyConnected::TernaryInputsInt8Weights::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kTernaryActTernaryWeight)
                ret = LowPrecision::FullyConnected::TernaryInputsTernaryWeights::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kBinaryActInt8Weight)
                ret = LowPrecision::FullyConnected::BinaryInputsInt8Weights::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kBinaryActBinaryWeight)
                ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeights::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kBinaryActBinaryWeightXOR)
                ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeightsXOR::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            else if (method &  LowPrecision::Method::kULPPACK)
                ret = LowPrecision::FullyConnected::ULPPACK::QuantizeFilter(input_ptr, input_padded_shape, output, layout);
            if (need_padding)
                LowPrecision::deallocate(input_ptr);
            return ret;
        }
        Status QuantizeInput(LowPrecision::Method method, const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout){
            int8_t* input_ptr = const_cast<int8_t*>(input);
            Shape input_padded_shape;
            input_padded_shape = GetPaddedShape(method, shape);
            bool need_padding = input_padded_shape != shape;
            if (need_padding){
                input_ptr = ::LowPrecision::allocate<int8_t>(input_padded_shape.flatsize);
                Status pad_ret = PadMatrixFromShapeToShape(input, input_ptr, shape, input_padded_shape);
                if (pad_ret != Status::Success) return pad_ret;
            }
            LowPrecision::Status ret = Status::NotSupported;
            if (method == LowPrecision::Method::kInt8Int4)
                ret = LowPrecision::FullyConnected::Int4::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kInt8Ternary)
                ret = LowPrecision::FullyConnected::Ternary::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kInt8QuaTernary)
                ret = LowPrecision::FullyConnected::Quaternary::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (
                method == LowPrecision::Method::kInt8Binary ||
                method == LowPrecision::Method::kFloat32Binary ||
                method == LowPrecision::Method::kFloat16Binary
            )
                ret = LowPrecision::FullyConnected::Binary::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kInt4ActInt8Weight)
                ret = LowPrecision::FullyConnected::Int4InputsInt8Weights::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kInt4ActInt4Weight)
                ret = LowPrecision::FullyConnected::Int4InputsInt4Weights::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kTernaryActInt8Weight)
                ret = LowPrecision::FullyConnected::TernaryInputsInt8Weights::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kTernaryActTernaryWeight)
                ret = LowPrecision::FullyConnected::TernaryInputsTernaryWeights::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kBinaryActInt8Weight)
                ret = LowPrecision::FullyConnected::BinaryInputsInt8Weights::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kBinaryActBinaryWeight)
                ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeights::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (method == LowPrecision::Method::kBinaryActBinaryWeightXOR)
                ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeightsXOR::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            else if (method &  LowPrecision::Method::kULPPACK)
                ret = LowPrecision::FullyConnected::ULPPACK::QuantizeInput(input_ptr, input_padded_shape, output, layout);
            if (need_padding)
                LowPrecision::deallocate(input_ptr);
            return ret;
        }
        Status Multiply(
            LowPrecision::Method method,
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape
        ){
            bool use_block_processing = GetVariableFromEnv( "UseBlockProcessing" ) == "TRUE";
            bool is_multibatched = input_shape.number_dims == 2 && input_shape.size[0] > 1;
            if (is_multibatched)
                if (use_block_processing)
                    return (Status)(MultiplyInt8MultiBatchedBlockProcessing(method, input, input_shape, kernel, kernel_shape, output, output_shape) | ((uint32_t)Status::MultiMultiplyBlock));
                else
                    return (Status)(MultiplyInt8MultiBatched(method, input, input_shape, kernel, kernel_shape, output, output_shape) | ((uint32_t)Status::MultiMultiply));
            else
                return (Status)(MultiplyInt8SingleBatch(method, input, input_shape, kernel, kernel_shape, output, output_shape) | ((uint32_t)Status::SingleMultiply));
        }
        Status MultiplyInt8SingleBatch(
            LowPrecision::Method method,
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape
        ){
            LowPrecision::Status ret;
            if (method == LowPrecision::Method::kInt4ActInt8Weight)
                ret = LowPrecision::FullyConnected::Int4InputsInt8Weights::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kInt4ActInt4Weight)
                ret = LowPrecision::FullyConnected::Int4InputsInt4Weights::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kTernaryActInt8Weight)
                ret = LowPrecision::FullyConnected::TernaryInputsInt8Weights::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kTernaryActTernaryWeight)
                ret = LowPrecision::FullyConnected::TernaryInputsTernaryWeights::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kBinaryActInt8Weight)
                ret = LowPrecision::FullyConnected::BinaryInputsInt8Weights::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kBinaryActBinaryWeight)
                ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeights::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kBinaryActBinaryWeightXOR)
                ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeightsXOR::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kInt8Int4)
                ret = LowPrecision::FullyConnected::Int4::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kInt8Binary)
                ret = LowPrecision::FullyConnected::Binary::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kInt8Ternary)
                ret = LowPrecision::FullyConnected::Ternary::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kInt8QuaTernary)
                ret = LowPrecision::FullyConnected::Quaternary::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kULPPACKW1A1)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    1, 1
                );
            else if (method == LowPrecision::Method::kULPPACKW2A2)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    2, 2
                );
            else if (method == LowPrecision::Method::kULPPACKW3A3)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    3, 3
                );
            else if (method == LowPrecision::Method::kULPPACKW4A4)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    4, 4
                );
            else if (method == LowPrecision::Method::kULPPACKW5A5)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    5, 5
                );
            else if (method == LowPrecision::Method::kULPPACKW6A6)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    6, 6
                );
            else if (method == LowPrecision::Method::kULPPACKW7A7)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8SingleBatch(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    7, 7
                );
            return ret;
        }
        Status MultiplyInt8MultiBatched(
            LowPrecision::Method method,
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape
        ){
            LowPrecision::Status ret;
            if (method == LowPrecision::Method::kInt4ActInt8Weight)
                ret = LowPrecision::FullyConnected::Int4InputsInt8Weights::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kInt4ActInt4Weight)
                ret = LowPrecision::FullyConnected::Int4InputsInt4Weights::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kTernaryActInt8Weight)
                ret = LowPrecision::FullyConnected::TernaryInputsInt8Weights::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kTernaryActTernaryWeight)
                ret = LowPrecision::FullyConnected::TernaryInputsTernaryWeights::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kBinaryActInt8Weight)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kBinaryActBinaryWeight)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kBinaryActBinaryWeightXOR)
                ret = Status::NotImplemented;
            else if (method == LowPrecision::Method::kULPPACKW1A1)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    1, 1
                );
            else if (method == LowPrecision::Method::kULPPACKW2A2)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    2, 2
                );
            else if (method == LowPrecision::Method::kULPPACKW3A3)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    3, 3
                );
            else if (method == LowPrecision::Method::kULPPACKW4A4)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    4, 4
                );
            else if (method == LowPrecision::Method::kULPPACKW5A5)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    5, 5
                );
            else if (method == LowPrecision::Method::kULPPACKW6A6)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    6, 6
                );
            else if (method == LowPrecision::Method::kULPPACKW7A7)
                ret = LowPrecision::FullyConnected::ULPPACK::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape,
                    7, 7
                );
            else if (method == LowPrecision::Method::kInt8Int4)
                ret = LowPrecision::FullyConnected::Int4::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kInt8Binary)
                ret = LowPrecision::FullyConnected::Binary::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            else if (method == LowPrecision::Method::kInt8Ternary)
                ret = LowPrecision::FullyConnected::Ternary::MultiplyInt8MultiBatched(
                    input, input_shape,
                    kernel, kernel_shape,
                    output, output_shape
                );
            // else if (method == LowPrecision::Method::kBinaryActInt8Weight)
            //         ret = LowPrecision::FullyConnected::BinaryInputsInt8Weights::MultiplyInt8MultiBatched(
            //             input, input_shape,
            //             kernel, kernel_shape,
            //             output, output_shape
            //         );
            // else if (method == LowPrecision::Method::kBinaryActBinaryWeight)
            //         ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeights::MultiplyInt8MultiBatched(
            //             input, input_shape,
            //             kernel, kernel_shape,
            //             output, output_shape
            //         );
            
            return ret;
        }
        Status MultiplyInt8MultiBatchedBlockProcessing(
            LowPrecision::Method method,
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape
        ){
            int input_stride_coeff  = 1,
                kernel_stride_coeff = 1, 
                output_stride_coeff = 4,
                block_num_batches   = 16,
                block_num_columns   = 64,
                block_num_rows      = 64,
                num_batches         = input_shape.size[0],
                num_columns         = input_shape.size[1],
                num_rows            = kernel_shape.size[1]
            ;
            if (GetVariableFromEnv( "BlockBatches" ) != "")
                block_num_batches = std::stoi(GetVariableFromEnv( "BlockBatches" ));
            if (GetVariableFromEnv( "BlockColumns" ) != "")
                block_num_columns = std::stoi(GetVariableFromEnv( "BlockColumns" ));
            if (GetVariableFromEnv( "BlockRows" )    != "")
                block_num_rows    = std::stoi(GetVariableFromEnv( "BlockRows" ));
            if (method == LowPrecision::Method::kInt4ActInt8Weight){
                input_stride_coeff  = 2;
                kernel_stride_coeff = 1;
                block_num_columns   = (block_num_columns%32)?(32):(block_num_columns);
                block_num_rows      = (block_num_rows%4)?(4):(block_num_rows);
            }
            else if (method == LowPrecision::Method::kInt4ActInt4Weight){
                input_stride_coeff  = 2;
                kernel_stride_coeff = 2;
                block_num_columns   = (block_num_columns%32)?(32):(block_num_columns);
                block_num_rows      = (block_num_rows%4)?(4):(block_num_rows);
            }
            else if (method == LowPrecision::Method::kTernaryActInt8Weight){
                input_stride_coeff  = 4;
                kernel_stride_coeff = 1;
                block_num_columns   = (block_num_columns%64)?(64):(block_num_columns);
                block_num_rows      = (block_num_rows%4)?(4):(block_num_rows);
            }
            else if (method == LowPrecision::Method::kTernaryActTernaryWeight){
                input_stride_coeff  = 4;
                kernel_stride_coeff = 4;
                block_num_columns   = (block_num_columns%64)?(64):(block_num_columns);
                block_num_rows      = (block_num_rows%4)?(4):(block_num_rows);
            }
            else if (method == LowPrecision::Method::kInt8Int4){
                input_stride_coeff  = 1;
                kernel_stride_coeff = 2;
                block_num_columns   = (block_num_columns%32)?(32):(block_num_columns);
                block_num_rows      = (block_num_rows%4)?(4):(block_num_rows);
            }
            else if (method == LowPrecision::Method::kInt8Binary){
                input_stride_coeff  = 1;
                kernel_stride_coeff = 8;
                block_num_columns   = (block_num_columns%128)?(128):(block_num_columns);
                block_num_rows      = (block_num_rows%4)?(4):(block_num_rows);
            }
            else if (method == LowPrecision::Method::kInt8Ternary){
                input_stride_coeff  = 1;
                kernel_stride_coeff = 4;
                block_num_columns   = (block_num_columns%64)?(64):(block_num_columns);
                block_num_rows      = (block_num_rows%4)?(4):(block_num_rows);
            }
            else
                return Status::NotImplemented;
            Params current_params;
            Status kernel_return_status;
            current_params.lhs_stride = num_columns / input_stride_coeff;
            current_params.rhs_stride = num_columns / input_stride_coeff;
            current_params.dst_stride = num_rows    *         4         ;

            int8_t*  input_c  = const_cast<int8_t*> (input);
            int8_t*  kernel_c = const_cast<int8_t*> (kernel);
            int32_t* output_c = const_cast<int32_t*>(output);

            for (size_t i = 0; i < num_batches; i += block_num_batches){
                current_params.start_batches         =    i    * block_num_batches;
                current_params.end_batches           = (i + 1) * block_num_batches;

                for (size_t j = 0; j < num_rows; j += block_num_rows){
                    current_params.start_rows        =    j    * block_num_rows;
                    current_params.end_rows          = (j + 1) * block_num_rows;

                    for (size_t k = 0; k < num_columns; k += block_num_columns){
                        current_params.start_columns =    k    * block_num_columns;
                        current_params.end_columns   = (k + 1) * block_num_columns;

                        int8_t*  lhs = input_c  + (i * num_columns) + k;
                        int8_t*  rhs = kernel_c + (j * num_columns) + k;
                        int32_t* dst = output_c + (i * num_rows)    + j;
                        
                        switch (method)
                        {
                        case LowPrecision::Method::kInt4ActInt8Weight:
                            kernel_return_status = LowPrecision::FullyConnected::Int4InputsInt8Weights::MultiplyInt8MultiBatchedBlock(
                                        lhs, rhs, dst, current_params
                            ); 
                            break;
                        case LowPrecision::Method::kInt4ActInt4Weight:
                            kernel_return_status = LowPrecision::FullyConnected::Int4InputsInt4Weights::MultiplyInt8MultiBatchedBlock(
                                        lhs, rhs, dst, current_params
                            ); 
                            break;
                        case LowPrecision::Method::kTernaryActInt8Weight:
                            kernel_return_status = LowPrecision::FullyConnected::TernaryInputsInt8Weights::MultiplyInt8MultiBatchedBlock(
                                        lhs, rhs, dst, current_params
                            ); 
                            break;
                        case LowPrecision::Method::kTernaryActTernaryWeight:
                            kernel_return_status = LowPrecision::FullyConnected::TernaryInputsTernaryWeights::MultiplyInt8MultiBatchedBlock(
                                        lhs, rhs, dst, current_params
                            ); 
                            break;
                        case LowPrecision::Method::kInt8Int4:
                            kernel_return_status = LowPrecision::FullyConnected::Int4::MultiplyInt8MultiBatchedBlock(
                                        lhs, rhs, dst, current_params
                            ); 
                            break;
                        case LowPrecision::Method::kInt8Binary:
                            kernel_return_status = LowPrecision::FullyConnected::Binary::MultiplyInt8MultiBatchedBlock(
                                        lhs, rhs, dst, current_params
                            ); 
                            break;
                        case LowPrecision::Method::kInt8Ternary:
                            kernel_return_status = LowPrecision::FullyConnected::Ternary::MultiplyInt8MultiBatchedBlock(
                                        lhs, rhs, dst, current_params
                            ); 
                            break;
                        default:
                            return Status::NotImplemented; 
                            break;
                        }
                        if ((LowPrecision::mask_out_source(kernel_return_status) != LowPrecision::Status::Success))
                            return kernel_return_status;
                    }
                }
            }
            return Status::Success;
        }
        Shape GetPaddedShape(const LowPrecision::Method method, const Shape& input_shape, bool pad_rows_too){
            int least_row_size = 4;
            int least_dim_size = 16;
            if (method == LowPrecision::Method::kInt4ActInt8Weight)
                least_dim_size = 32;
            else if (method == LowPrecision::Method::kInt4ActInt4Weight)
                least_dim_size = 32;
            else if (method == LowPrecision::Method::kTernaryActInt8Weight)
                least_dim_size = 64;
            else if (method == LowPrecision::Method::kTernaryActTernaryWeight)
                least_dim_size = 64;
            else if (method == LowPrecision::Method::kBinaryActInt8Weight)
                least_dim_size = 128;
            else if (method == LowPrecision::Method::kBinaryActBinaryWeight)
                least_dim_size = 128;
            else if (method == LowPrecision::Method::kBinaryActBinaryWeightXOR)
                least_dim_size = 128;
            else if (method == LowPrecision::Method::kInt8Int4)
                least_dim_size = 32;
            else if (method == LowPrecision::Method::kInt8Binary)
                least_dim_size = 128;
            else if (method == LowPrecision::Method::kInt8Ternary)
                least_dim_size = 64;
            else if (method == LowPrecision::Method::kInt8QuaTernary)
                least_dim_size = 64;
            if (input_shape.number_dims == 1){
                int padding_size = (input_shape.size[0] % least_dim_size)?(least_dim_size - (input_shape.size[0] % least_dim_size)):(0);
                Shape new_shape;
                new_shape.number_dims = input_shape.number_dims;
                new_shape.size = new int[new_shape.number_dims];
                new_shape.size[0] = ::ceil(input_shape.size[0] / ((float)least_dim_size)) * least_dim_size;
                new_shape.flatsize = ::LowPrecision::FullyConnected::CalcFlatSize(new_shape.size, 1);
                return new_shape;
            }
            int padding_size = (input_shape.size[1] % least_dim_size)?(least_dim_size - (input_shape.size[1] % least_dim_size)):(0);
            Shape new_shape;
            new_shape.number_dims = input_shape.number_dims;
            new_shape.size = new int[new_shape.number_dims];
            if (pad_rows_too)
                new_shape.size[0] = ::ceil(input_shape.size[0] / ((float)least_row_size)) * least_row_size;
            else
                new_shape.size[0] = input_shape.size[0];
            new_shape.size[1] = ::ceil(input_shape.size[1] / ((float)least_dim_size)) * least_dim_size;
            new_shape.flatsize = ::LowPrecision::FullyConnected::CalcFlatSize(new_shape.size, 2);
            return new_shape;
        }
        Status PadMatrixFromShapeToShape(const int8_t* input, int8_t* output, Shape from_shape, Shape to_shape, const int8_t pad_value){
            if (from_shape.number_dims != to_shape.number_dims) return Status::DimensionsMisMatch;
            if (from_shape.number_dims <= 0) return Status::SizesMisMatch;
            int num_dims = from_shape.number_dims;
            if (num_dims == 2){ // We only accept matrix or vector.
                for (int j = 0; j < from_shape.size[0]; j++){
                    std::copy(&input[j * from_shape.size[1]], &input[(j + 1) * from_shape.size[1]], &output[j * to_shape.size[1]]);
                    for (int i = from_shape.size[1]; i < to_shape.size[1]; i++)
                        output[j * to_shape.size[1] + i] = pad_value;
                }
                for (int j = from_shape.size[0]; j < to_shape.size[0]; j++){
                    for (int i = 0; i < to_shape.size[1]; i++)
                        output[j * to_shape.size[1] + i] = pad_value;
                }
            }
            else{
                std::copy(input, &input[from_shape.size[0]], output);
                for (int i = from_shape.size[0]; i < to_shape.size[0]; i++)
                    output[i] = pad_value;
            }
            return Status::Success;
        }
        template<typename T>
        Status DePadMatrixFromShapeToShape(const T* input, T* output, Shape from_shape, Shape to_shape){
            if (from_shape.number_dims < to_shape.number_dims) return Status::DimensionsMisMatch;
            if (from_shape.number_dims <= 0) return Status::SizesMisMatch;
            int num_dims = to_shape.number_dims;
            // std::cout << "DePadMatrixFromShapeToShape("
            //             << get_shape_string(from_shape) << ", "
            //             << get_shape_string(to_shape) << ")"
            //             << std::endl;
            if (num_dims == 2){ // We only accept matrix or vector.
                for (int j = 0; j < to_shape.size[0]; j++){
                    std::copy(&input[j * from_shape.size[1]], &input[j * from_shape.size[1] + to_shape.size[1]], &output[j * to_shape.size[1]]);
                }
            }
            else{
                std::copy(input, &input[to_shape.size[0]], output);
            }
            return Status::Success;
        }
        Status ApplyDowncast(int32_t* input, int8_t* output, Shape shape, const int32_t downcast_coeff){
            if (shape.number_dims == 1){
                for (int i = 0; i < shape.size[0]; i++)
                    output[i] = input[i] / downcast_coeff;
            }
            else if (shape.number_dims == 2){
                for (int j = 0; j < shape.size[0]; j++)
                    for (int i = 0; i < shape.size[1]; i++)
                        output[j * shape.size[1] + i] = input[j * shape.size[1] + i] / downcast_coeff;
            }
            else
                return Status::NotSupported;
            return Status::Success;
        }
        void doScallingFactorMultiplication(int32_t* input, const float* scalling_factor, float* output,
                                            int batch_n, int input_n){
            for(int i = 0 ; i < batch_n ; i++)
                for(int j = 0 ; j < input_n ; j++)
                    output[i * input_n +  j] = input[i * input_n +  j] * scalling_factor[j];
            return;
        }
        
        Status Mul(Matrix& lhs, Matrix& rhs, Matrix& dst, Method method){
            // std::cout << "Mul: " << std::endl;
            //     std::cout << "\tLHS:{ ";
            //         std::cout << "Shape: ";
            //         std::cout << LowPrecision::get_shape_string(lhs.getShape());
            //         std::cout << " NeedScratchpad: ";
            //         std::cout << lhs.getNeedScratchpad();
            //         std::cout << " isScratchpadValid: ";
            //         std::cout << lhs.isScratchpadValid();
            //         std::cout << " Date: ";
            //         std::cout << LowPrecision::get_pointer_as<void>(lhs.getData());
            //     std::cout << " }";
            // std::cout << std::endl;
            //     std::cout << "\tRHS:{ ";
            //         std::cout << "Shape: ";
            //         std::cout << LowPrecision::get_shape_string(rhs.getShape());
            //         std::cout << " NeedScratchpad: ";
            //         std::cout << rhs.getNeedScratchpad();
            //         std::cout << " isScratchpadValid: ";
            //         std::cout << rhs.isScratchpadValid();
            //         std::cout << " Date: ";
            //         std::cout << LowPrecision::get_pointer_as<void>(rhs.getData());
            //     std::cout << " }";
            // std::cout << std::endl;
            //     std::cout << "\tDST:{ ";
            //         std::cout << "Shape: ";
            //         std::cout << LowPrecision::get_shape_string(dst.getShape());
            //         std::cout << " NeedScratchpad: ";
            //         std::cout << dst.getNeedScratchpad();
            //         std::cout << " isScratchpadValid: ";
            //         std::cout << dst.isScratchpadValid();
            //         std::cout << " Date: ";
            //         std::cout << LowPrecision::get_pointer_as<void>(dst.getData());
            //     std::cout << " }";
            // std::cout << std::endl;
            if (lhs.getNeedScratchpad() && !lhs.isScratchpadValid() && lhs.getData() == nullptr)
                return (Status)(((uint32_t)Status::LHSNotInitialized) | ((uint32_t)Status::MulAPI));
            if (rhs.getNeedScratchpad() && !rhs.isScratchpadValid() && rhs.getData() == nullptr)
                return (Status)(((uint32_t)Status::RHSNotInitialized) | ((uint32_t)Status::MulAPI));
            if (dst.getNeedScratchpad() && !dst.isScratchpadValid() && dst.getData() == nullptr)
                return (Status)(((uint32_t)Status::DSTNotInitialized) | ((uint32_t)Status::MulAPI));
            if (dst.getNeedScratchpad() && dst.getNeedDowncast())
                return (Status)(((uint32_t)Status::NeedDowncastWScratch) | ((uint32_t)Status::MulAPI));

            // Check if the data is in scratchpad.
            // If not, process the data and put it in scratchpad.
            // If so,  continue to process from scratchpad.

            // std::cout << "Sizes:\n";
            // std::cout << "\tInput:" << LowPrecision::get_shape_string(lhs.getShape()) << "\n";
            // std::cout << "\tKernel:" << LowPrecision::get_shape_string(rhs.getShape()) << "\n";
            // std::cout << "\tOutput:" << LowPrecision::get_shape_string(dst.getShape()) << "\n";
            
            if (lhs.getNeedScratchpad() && !lhs.isScratchpadValid()){
                Status input_ret = QuantizeInput(method, lhs.getData(), lhs.getShape(), lhs.getScratchpad(), lhs.getMemLayout());
                if (input_ret == Status::NotNeeded)
                    lhs.setNeedScratchpad(false);
                else if (input_ret != Status::Success)
                    return (Status)(input_ret | ((uint32_t)Status::InputQuantizition));
                else
                    lhs.setScratchpadValid();
            }
            if (rhs.getNeedScratchpad() && !rhs.isScratchpadValid()){
                Status input_ret = QuantizeFilter(method, rhs.getData(), rhs.getShape(), rhs.getScratchpad(), rhs.getMemLayout());
                if (input_ret != Status::Success)
                    return (Status)(input_ret | ((uint32_t)Status::FilterQuantizition));
                rhs.setScratchpadValid();
            }

            int8_t*  lhs_p =                         (lhs.getNeedScratchpad())?(lhs.getScratchpad()):(lhs.getData());
            int8_t*  rhs_p =                         (rhs.getNeedScratchpad())?(rhs.getScratchpad()):(rhs.getData());
            int32_t* dst_p = get_pointer_as<int32_t>((dst.getNeedScratchpad())?(dst.getScratchpad()):(dst.getData()));
            bool     need_downcast = dst.getNeedDowncast();

            if (need_downcast)
                dst_p = LowPrecision::allocate<int32_t>(dst.getShape().flatsize);
                
            int num_batches = 1;
            if (lhs.getShape().number_dims >= 2)
                num_batches = lhs.getShape().size[0];
            LowPrecision::Status mul_ret_status;
            if (num_batches > 1 && num_batches % 4 && !(method & LowPrecision::Method::kULPPACK)){
                int32_t* dst_p_backup = nullptr;
                Shape dst_shape, lhs_shape;
                dst_shape = dst.getShape();
                lhs_shape = lhs.getShape();
                // std::cout << "From Shape " 
                //           << LowPrecision::get_shape_string(dst_shape);
                dst_shape.size[0] = ::ceil(dst_shape.size[0] / 4.0) * 4;
                lhs_shape.size[0] = ::ceil(lhs_shape.size[0] / 4.0) * 4;
                dst_shape.flatsize = CalcFlatSize(dst_shape.size, dst_shape.number_dims);
                lhs_shape.flatsize = CalcFlatSize(lhs_shape.size, lhs_shape.number_dims);
                // std::cout << " To Shape " 
                //           << LowPrecision::get_shape_string(dst_shape) << std::endl;
                // std::cout << "\tDST:{ ";
                //     std::cout << "Shape: ";
                //     std::cout << LowPrecision::get_shape_string(dst_shape);
                //     std::cout << " NeedScratchpad: ";
                //     std::cout << dst.getNeedScratchpad();
                //     std::cout << " isScratchpadValid: ";
                //     std::cout << dst.isScratchpadValid();
                //     std::cout << " Date: ";
                //     std::cout << LowPrecision::get_pointer_as<void>(dst.getData());
                // std::cout << " } (Changed)";
                // std::cout << std::endl;
                // std::cout << "\tLHS:{ ";
                //     std::cout << "Shape: ";
                //     std::cout << LowPrecision::get_shape_string(lhs_shape);
                //     std::cout << " NeedScratchpad: ";
                //     std::cout << lhs.getNeedScratchpad();
                //     std::cout << " isScratchpadValid: ";
                //     std::cout << lhs.isScratchpadValid();
                //     std::cout << " Date: ";
                //     std::cout << LowPrecision::get_pointer_as<void>(lhs.getData());
                // std::cout << " } (Changed)";
                // std::cout << std::endl;
                dst_p_backup = LowPrecision::allocate<int32_t>(dst_shape.flatsize);

                mul_ret_status = Multiply(method, lhs_p, lhs_shape, rhs_p, rhs.getShape(), dst_p_backup, dst.getShape());

                DePadMatrixFromShapeToShape(dst_p_backup, dst_p, dst_shape, dst.getShape());
                
                LowPrecision::deallocate(dst_p_backup, false);
            } else if (num_batches % 8 && method & LowPrecision::Method::kULPPACK){
                int32_t* dst_p_backup = nullptr;
                Shape dst_shape, lhs_shape;
                dst_shape = dst.getShape();
                lhs_shape = lhs.getShape();
                if (dst_shape.number_dims < 2)
                    dst_shape.extend_dims();
                if (lhs_shape.number_dims < 2)
                    lhs_shape.extend_dims();
                // std::cout << "From Shape " 
                //           << LowPrecision::get_shape_string(dst_shape);
                dst_shape.size[dst_shape.number_dims - 2] = ::ceil(dst_shape.size[dst_shape.number_dims - 2] / 8.0) * 8;
                lhs_shape.size[lhs_shape.number_dims - 2] = ::ceil(lhs_shape.size[lhs_shape.number_dims - 2] / 8.0) * 8;
                dst_shape.flatsize = CalcFlatSize(dst_shape.size, dst_shape.number_dims);
                lhs_shape.flatsize = CalcFlatSize(lhs_shape.size, lhs_shape.number_dims);
                // std::cout << " To Shape " 
                //           << LowPrecision::get_shape_string(dst_shape) << std::endl;

                // std::cout << "\tDST:{ ";
                //     std::cout << "Shape: ";
                //     std::cout << LowPrecision::get_shape_string(dst_shape);
                //     std::cout << " NeedScratchpad: ";
                //     std::cout << dst.getNeedScratchpad();
                //     std::cout << " isScratchpadValid: ";
                //     std::cout << dst.isScratchpadValid();
                //     std::cout << " Date: ";
                //     std::cout << LowPrecision::get_pointer_as<void>(dst.getData());
                // std::cout << " } (Changed)";
                // std::cout << std::endl;
                // std::cout << "\tLHS:{ ";
                //     std::cout << "Shape: ";
                //     std::cout << LowPrecision::get_shape_string(lhs_shape);
                //     std::cout << " NeedScratchpad: ";
                //     std::cout << lhs.getNeedScratchpad();
                //     std::cout << " isScratchpadValid: ";
                //     std::cout << lhs.isScratchpadValid();
                //     std::cout << " Date: ";
                //     std::cout << LowPrecision::get_pointer_as<void>(lhs.getData());
                // std::cout << " } (Changed)";
                // std::cout << std::endl;
                dst_p_backup = LowPrecision::allocate<int32_t>(dst_shape.flatsize);

                mul_ret_status = Multiply(method, lhs_p, lhs_shape, rhs_p, rhs.getShape(), dst_p_backup, dst_shape);

                Status depad_ret = DePadMatrixFromShapeToShape(dst_p_backup, dst_p, dst_shape, dst.getShape());
                if (depad_ret != Status::Success)
                    return (Status)(depad_ret | ((uint32_t)Status::DepadMatrix));
                LowPrecision::deallocate(dst_p_backup, false);
            } else {
                mul_ret_status = Multiply(method, lhs_p, lhs.getShape(), rhs_p, rhs.getShape(), dst_p, dst.getShape());
            }

            if (need_downcast){
                Status downcast_ret = ApplyDowncast(dst_p, dst.getData(), dst.getShape(), dst.getDowncastCoeff());
                if (downcast_ret != Status::Success)
                    return (Status)(downcast_ret | ((uint32_t)Status::ApplyDowncast));
                LowPrecision::deallocate(dst_p);
            }
            
            return mul_ret_status;
        }
    }
}
#endif
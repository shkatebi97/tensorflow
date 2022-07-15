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
            else if (retval == std::string("I3-I3"))
                return Method::kInt3ActInt3Weight;
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
            if (!Is_FC)
                return false;
            bool multibatched_enabled = !(GetVariableFromEnv("LowPrecisionMultiBatched") == "FALSE");
            bool is_multibatched = input_shape.number_dims == 2 && input_shape.size[0] > 1; 
            // Checking for Not-Supported Input DataTypes
            if (
                (input_type != DataType::Int8 && input_type != DataType::Float32 && input_type != DataType::Int32) ||
                filter_type != DataType::Int8 ||
                (output_type != DataType::Float32 && output_type != DataType::Int32))
                return false;
            // Checking for the conditions of rejection of multi-batched input
            if(!multibatched_enabled && is_multibatched)
                return false;
            if (
                is_multibatched && 
                multibatched_enabled &&
                input_shape.size[0] % 4
            )
                return false;

            // Checking common conditions
            if (filter_shape.size[input_shape.number_dims - 2] % 4)
                return false;

            // Checking conditions of input shape of any method
            if (method == Method::kInt8Binary ||
                method == Method::kFloat32Binary ||
                method == Method::kFloat16Binary)
                return !(input_shape.size[input_shape.number_dims - 1] % 128);
            if (method == Method::kInt8Ternary ||
                method == Method::kFloat32Ternary ||
                method == Method::kFloat16Ternary ||
                method == Method::kInt8QuaTernary)
                return !(input_shape.size[input_shape.number_dims - 1] % 64);
            if (method == Method::kInt8Int4)
                return !(input_shape.size[input_shape.number_dims - 1] % 32);
            if (method == Method::kInt4ActInt8Weight)
                return !(input_shape.size[input_shape.number_dims - 1] % 32);
            if (method == Method::kInt4ActInt4Weight)
                return !(input_shape.size[input_shape.number_dims - 1] % 32);
            if (method == Method::kTernaryActInt8Weight)
                return !(input_shape.size[input_shape.number_dims - 1] % 64);
            if (method == Method::kTernaryActTernaryWeight)
                return !(input_shape.size[input_shape.number_dims - 1] % 64);
            if (method == Method::kBinaryActInt8Weight)
                return !(input_shape.size[input_shape.number_dims - 1] % 128);
            if (method == Method::kBinaryActBinaryWeight)
                return !(input_shape.size[input_shape.number_dims - 1] % 128);
            if (method == Method::kInt3ActInt3Weight)
                return !(input_shape.size[input_shape.number_dims - 1] % 40);
            // If none of the aboves
            return false;
        }
        bool IncludesActivationCompression(Method method){
            return 
                (method & Method::kInt4ActInt8Weight)       || 
                (method & Method::kInt4ActInt4Weight)       ||
                (method & Method::kTernaryActInt8Weight)    ||
                (method & Method::kTernaryActTernaryWeight) ||
                (method & Method::kBinaryActInt8Weight)     ||
                (method & Method::kBinaryActBinaryWeight)   ||
                (method & Method::kInt3ActInt3Weight)
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
            if (method == LowPrecision::Method::kInt8Int4)
                return LowPrecision::FullyConnected::Int4::TransformFilterShape(shape, n_dims);
            else if (method == LowPrecision::Method::kInt8Ternary)
                return LowPrecision::FullyConnected::Ternary::TransformFilterShape(shape, n_dims);
            else if (method == LowPrecision::Method::kInt8QuaTernary)
                return LowPrecision::FullyConnected::Quaternary::TransformFilterShape(shape, n_dims);
            else if (
            method == LowPrecision::Method::kInt8Binary ||
            method == LowPrecision::Method::kFloat32Binary ||
            method == LowPrecision::Method::kFloat16Binary
            )
                return LowPrecision::FullyConnected::Binary::TransformFilterShape(shape, n_dims);
            else if ( method == LowPrecision::Method::kInt4ActInt8Weight )
                return LowPrecision::FullyConnected::Int4InputsInt8Weights::TransformFilterShape(shape, n_dims);
            else if ( method == LowPrecision::Method::kInt4ActInt4Weight )
                return LowPrecision::FullyConnected::Int4InputsInt4Weights::TransformFilterShape(shape, n_dims);
            else if ( method == LowPrecision::Method::kTernaryActInt8Weight )
                return LowPrecision::FullyConnected::TernaryInputsInt8Weights::TransformFilterShape(shape, n_dims);
            else if ( method == LowPrecision::Method::kTernaryActTernaryWeight )
                return LowPrecision::FullyConnected::TernaryInputsTernaryWeights::TransformFilterShape(shape, n_dims);
            else if ( method == LowPrecision::Method::kBinaryActInt8Weight )
                return LowPrecision::FullyConnected::BinaryInputsInt8Weights::TransformFilterShape(shape, n_dims);
            else if ( method == LowPrecision::Method::kBinaryActBinaryWeight )
                return LowPrecision::FullyConnected::BinaryInputsBinaryWeights::TransformFilterShape(shape, n_dims);
            return 0;
        }
        size_t TransformInputShape(LowPrecision::Method method, int* shape, int n_dims){
            if (method == LowPrecision::Method::kInt8Int4)
                return LowPrecision::FullyConnected::Int4::TransformInputShape(shape, n_dims);
            else if (method == LowPrecision::Method::kInt8Ternary)
                return LowPrecision::FullyConnected::Ternary::TransformInputShape(shape, n_dims);
            else if (method == LowPrecision::Method::kInt8QuaTernary)
                return LowPrecision::FullyConnected::Quaternary::TransformInputShape(shape, n_dims);
            else if (
            method == LowPrecision::Method::kInt8Binary ||
            method == LowPrecision::Method::kFloat32Binary ||
            method == LowPrecision::Method::kFloat16Binary
            )
                return LowPrecision::FullyConnected::Binary::TransformInputShape(shape, n_dims);
            else if (method == LowPrecision::Method::kInt4ActInt8Weight)
                return LowPrecision::FullyConnected::Int4InputsInt8Weights::TransformInputShape(shape, n_dims);
            else if (method == LowPrecision::Method::kInt4ActInt4Weight)
                return LowPrecision::FullyConnected::Int4InputsInt4Weights::TransformInputShape(shape, n_dims);
            else if (method == LowPrecision::Method::kTernaryActInt8Weight)
                return LowPrecision::FullyConnected::TernaryInputsInt8Weights::TransformInputShape(shape, n_dims);
            else if (method == LowPrecision::Method::kTernaryActTernaryWeight)
                return LowPrecision::FullyConnected::TernaryInputsTernaryWeights::TransformInputShape(shape, n_dims);
            else if (method == LowPrecision::Method::kBinaryActInt8Weight)
                return LowPrecision::FullyConnected::BinaryInputsInt8Weights::TransformInputShape(shape, n_dims);
            else if (method == LowPrecision::Method::kBinaryActBinaryWeight)
                return LowPrecision::FullyConnected::BinaryInputsBinaryWeights::TransformInputShape(shape, n_dims);
            return 0;
        }
        Status QuantizeFilter(LowPrecision::Method method, const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout){
            LowPrecision::Status ret;
            if (method == LowPrecision::Method::kInt8Int4)
                ret = LowPrecision::FullyConnected::Int4::QuantizeFilter(input, k_shape, output, layout);
            else if (method == LowPrecision::Method::kInt8Ternary)
                ret = LowPrecision::FullyConnected::Ternary::QuantizeFilter(input, k_shape, output, layout);
            else if (method == LowPrecision::Method::kInt8QuaTernary)
                ret = LowPrecision::FullyConnected::Quaternary::QuantizeFilter(input, k_shape, output, layout);
            else if (
                method == LowPrecision::Method::kInt8Binary ||
                method == LowPrecision::Method::kFloat32Binary ||
                method == LowPrecision::Method::kFloat16Binary
            )
                ret = LowPrecision::FullyConnected::Binary::QuantizeFilter(input, k_shape, output, layout);
            else if (method == LowPrecision::Method::kInt4ActInt8Weight)
                ret = LowPrecision::FullyConnected::Int4InputsInt8Weights::QuantizeFilter(input, k_shape, output, layout);
            else if (method == LowPrecision::Method::kInt4ActInt4Weight)
                ret = LowPrecision::FullyConnected::Int4InputsInt4Weights::QuantizeFilter(input, k_shape, output, layout);
            else if (method == LowPrecision::Method::kTernaryActInt8Weight)
                ret = LowPrecision::FullyConnected::TernaryInputsInt8Weights::QuantizeFilter(input, k_shape, output, layout);
            else if (method == LowPrecision::Method::kTernaryActTernaryWeight)
                ret = LowPrecision::FullyConnected::TernaryInputsTernaryWeights::QuantizeFilter(input, k_shape, output, layout);
            else if (method == LowPrecision::Method::kBinaryActInt8Weight)
                ret = LowPrecision::FullyConnected::BinaryInputsInt8Weights::QuantizeFilter(input, k_shape, output, layout);
            else if (method == LowPrecision::Method::kBinaryActBinaryWeight)
                ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeights::QuantizeFilter(input, k_shape, output, layout);
            return ret;
        }
        Status QuantizeInput(LowPrecision::Method method, const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout){
            LowPrecision::Status ret;
            if (method == LowPrecision::Method::kInt8Int4)
                ret = LowPrecision::FullyConnected::Int4::QuantizeInput(input, shape, output, layout);
            else if (method == LowPrecision::Method::kInt8Ternary)
                ret = LowPrecision::FullyConnected::Ternary::QuantizeInput(input, shape, output, layout);
            else if (method == LowPrecision::Method::kInt8QuaTernary)
                ret = LowPrecision::FullyConnected::Quaternary::QuantizeInput(input, shape, output, layout);
            else if (
                method == LowPrecision::Method::kInt8Binary ||
                method == LowPrecision::Method::kFloat32Binary ||
                method == LowPrecision::Method::kFloat16Binary
            )
                ret = LowPrecision::FullyConnected::Binary::QuantizeInput(input, shape, output, layout);
            else if (method == LowPrecision::Method::kInt4ActInt8Weight)
                ret = LowPrecision::FullyConnected::Int4InputsInt8Weights::QuantizeInput(input, shape, output, layout);
            else if (method == LowPrecision::Method::kInt4ActInt4Weight)
                ret = LowPrecision::FullyConnected::Int4InputsInt4Weights::QuantizeInput(input, shape, output, layout);
            else if (method == LowPrecision::Method::kTernaryActInt8Weight)
                ret = LowPrecision::FullyConnected::TernaryInputsInt8Weights::QuantizeInput(input, shape, output, layout);
            else if (method == LowPrecision::Method::kTernaryActTernaryWeight)
                ret = LowPrecision::FullyConnected::TernaryInputsTernaryWeights::QuantizeInput(input, shape, output, layout);
            else if (method == LowPrecision::Method::kBinaryActInt8Weight)
                ret = LowPrecision::FullyConnected::BinaryInputsInt8Weights::QuantizeInput(input, shape, output, layout);
            else if (method == LowPrecision::Method::kBinaryActBinaryWeight)
                ret = LowPrecision::FullyConnected::BinaryInputsBinaryWeights::QuantizeInput(input, shape, output, layout);
            return ret;
        }
        Status Multiply(
            LowPrecision::Method method,
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape){
            bool is_multibatched = input_shape.number_dims == 2 && input_shape.size[0] > 1;
            if (is_multibatched)
                return (Status)(MultiplyInt8MultiBatched(method, input, input_shape, kernel, kernel_shape, output, output_shape) | ((uint32_t)Status::MultiMultiply));
            else
                return (Status)(MultiplyInt8SingleBatch(method, input, input_shape, kernel, kernel_shape, output, output_shape) | ((uint32_t)Status::SingleMultiply));
        }
        Status MultiplyInt8SingleBatch(
            LowPrecision::Method method,
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape){
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
                return ret;
            }
        Status MultiplyInt8MultiBatched(
            LowPrecision::Method method,
            const int8_t* input, LowPrecision::Shape input_shape,
            const int8_t* kernel, LowPrecision::Shape kernel_shape,
            int32_t* output, LowPrecision::Shape output_shape){
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

        void doScallingFactorMultiplication(int32_t* input, const float* scalling_factor, float* output,
                                            int batch_n, int input_n){
            for(int i = 0 ; i < batch_n ; i++)
                for(int j = 0 ; j < input_n ; j++)
                    output[i * input_n +  j] = input[i * input_n +  j] * scalling_factor[j];
            return;
        }
        
        Status Mul(Matrix& lhs, Matrix& rhs, Matrix& dst, Method method){
            if (lhs.getNeedScratchpad() && !lhs.isScratchpadValid() && lhs.getData() == nullptr)
                return (Status)(((uint32_t)Status::LHSNotInitialized) | ((uint32_t)Status::Mul));
            if (rhs.getNeedScratchpad() && !rhs.isScratchpadValid() && rhs.getData() == nullptr)
                return (Status)(((uint32_t)Status::RHSNotInitialized) | ((uint32_t)Status::Mul));
            if (dst.getNeedScratchpad() && !dst.isScratchpadValid() && dst.getData() == nullptr)
                return (Status)(((uint32_t)Status::DSTNotInitialized) | ((uint32_t)Status::Mul));
            // Check if the data is in scratchpad.
            // If not, process the data and put it in scratchpad.
            // If so,  continue to process from scratchpad.
            if (lhs.getNeedScratchpad() && !lhs.isScratchpadValid()){
                Status input_ret = QuantizeInput(method, lhs.getData(), lhs.getShape(), lhs.getScratchpad(), lhs.getMemLayout());
                if (input_ret != Status::Success)
                    return (Status)(input_ret | ((uint32_t)Status::InputQuantizition));
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

            return Multiply(method, lhs_p, lhs.getShape(), rhs_p, rhs.getShape(), dst_p, dst.getShape());
        }
    }
}
#endif
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
        Method GetMethodFromEnv(){
            char * val;
            val = getenv( "LowPrecisionFC" );
            std::string retval = "";                                                          
            if (val != NULL) {
                retval = val;
            }
            if (retval == std::string("I8-I4"))
                return Method::kInt8Int4;
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
            Method method, Shape input_shape, 
            DataType input_type, DataType filter_type, DataType output_type){
            if (
                (input_type != DataType::Int8 && input_type != DataType::Float32 && input_type != DataType::Int32) ||
                filter_type != DataType::Int8 ||
                (output_type != DataType::Float32 && output_type != DataType::Int32))
                return false;
            if (input_shape.size[0] != 1)
                return false;
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
            return false;
        }
        
        Status QuantizeFilterToInt4(const int8_t* input, Shape k_shape, int8_t* output, MemLayout layout){
            if (k_shape.number_dims != 2)
                return Status::DimensionsMisMatch;
            if (k_shape.size[1] % 32)
                return Status::SizesMisMatch;
            int new_weights_length = (k_shape.size[1] / 2) * k_shape.size[0];
            int8_t* temp = LowPrecision::allocate<int8_t>(new_weights_length);
            uint8_t* temp_u = get_pointer_as<uint8_t>(temp);
            #ifdef PRINT_VALUES
            std::vector<std::tuple<int, int, int, int, int, int, int, int, int, int>> memory_accesses;
            std::cout << get_shape_string(k_shape) << std::endl;
            #endif
            if (layout == MemLayout::kColumnMajor)
                for (int i = 0 ; i < k_shape.size[0] ; i++){
                    for (int j = 0 ; j < k_shape.size[1] ; j++){
                        int cluster_id = j / 32,
                            container_id = (j % 32) % 16,
                            shift_amount = ((j % 32) / 16);
                        temp_u[i * (k_shape.size[1]/2) + ((cluster_id * 16) + container_id)] |=
                            quantizeTo4BitIntAndPackBitsStep(input[j * k_shape.size[0] + i], shift_amount);
                        #ifdef PRINT_VALUES
                        std::tuple<int, int, int, int, int, int, int, int, int, int> access;
                        std::get<0>(access) = i;
                        std::get<1>(access) = j;
                        std::get<2>(access) = i * (k_shape.size[1]/2) + ((cluster_id * 16) + container_id);
                        std::get<3>(access) = shift_amount * 4;
                        std::get<4>(access) = cluster_id;
                        std::get<5>(access) = container_id;
                        std::get<6>(access) = j * k_shape.size[0] + i;
                        std::get<7>(access) = input[j * k_shape.size[0] + i];
                        std::get<8>(access) = 0x0f << ((shift_amount) * 4);
                        std::get<9>(access) = (input[j * k_shape.size[0] + i] << (shift_amount * 4)) & (0x0f << ((shift_amount) * 4));
                        memory_accesses.push_back(access);
                        #endif
                    }
                }
            else
                for (int i = 0 ; i < k_shape.size[0] ; i++){
                    for (int j = 0 ; j < k_shape.size[1] ; j++){
                        int cluster_id = j / 32,
                            container_id = (j % 32) % 16,
                            shift_amount = ((j % 32) / 16);
                        temp_u[i * (k_shape.size[1]/2) + ((cluster_id * 16) + container_id)] |=
                            quantizeTo4BitIntAndPackBitsStep(input[i * k_shape.size[1] + j], shift_amount);
                        #ifdef PRINT_VALUES
                        std::tuple<int, int, int, int, int, int, int, int, int, int> access;
                        std::get<0>(access) = i;
                        std::get<1>(access) = j;
                        std::get<2>(access) = i * (k_shape.size[1]/2) + ((cluster_id * 16) + container_id);
                        std::get<3>(access) = shift_amount * 4;
                        std::get<4>(access) = cluster_id;
                        std::get<5>(access) = container_id;
                        std::get<6>(access) = i * k_shape.size[1] + j;
                        std::get<7>(access) = input[i * k_shape.size[1] + j];
                        std::get<8>(access) = 0x0f << ((shift_amount) * 4);
                        std::get<9>(access) = (input[i * k_shape.size[1] + j] << (shift_amount * 4)) & (0x0f << ((shift_amount) * 4));
                        memory_accesses.push_back(access);
                        #endif
                    }
                }
            #ifdef PRINT_VALUES
            for (int i = 0 ; i < memory_accesses.size() ; i++)
                std::cout << "\t( " 
                          << ((std::get<0>(memory_accesses[i]) < 100)?(" "):("")) << ((std::get<0>(memory_accesses[i]) < 10 && std::get<0>(memory_accesses[i]) >= 0)?(" "):("")) << std::get<0>(memory_accesses[i]) << ", "
                          << ((std::get<1>(memory_accesses[i]) < 100)?(" "):("")) << ((std::get<1>(memory_accesses[i]) < 10 && std::get<1>(memory_accesses[i]) >= 0)?(" "):("")) << std::get<1>(memory_accesses[i]) << ", "
                          << ((std::get<2>(memory_accesses[i]) < 100)?(" "):("")) << ((std::get<2>(memory_accesses[i]) < 10 && std::get<2>(memory_accesses[i]) >= 0)?(" "):("")) << std::get<2>(memory_accesses[i]) << ", "
                          << ((std::get<3>(memory_accesses[i]) < 100)?(" "):("")) << ((std::get<3>(memory_accesses[i]) < 10 && std::get<3>(memory_accesses[i]) >= 0)?(" "):("")) << std::get<3>(memory_accesses[i]) << ", ("
                          << ((std::get<4>(memory_accesses[i]) < 100)?(" "):("")) << ((std::get<4>(memory_accesses[i]) < 10 && std::get<4>(memory_accesses[i]) >= 0)?(" "):("")) << std::get<4>(memory_accesses[i]) << ", "
                          << ((std::get<5>(memory_accesses[i]) < 100)?(" "):("")) << ((std::get<5>(memory_accesses[i]) < 10 && std::get<5>(memory_accesses[i]) >= 0)?(" "):("")) << std::get<5>(memory_accesses[i]) << "), "
                          << ((std::get<6>(memory_accesses[i]) < 100)?(" "):("")) << ((std::get<6>(memory_accesses[i]) < 10 && std::get<6>(memory_accesses[i]) >= 0)?(" "):("")) << std::get<6>(memory_accesses[i]) << ", "
                          << ((std::get<7>(memory_accesses[i]) < 100)?(" "):("")) << ((std::get<7>(memory_accesses[i]) < 10 && std::get<7>(memory_accesses[i]) >= 0)?(" "):("")) << std::get<7>(memory_accesses[i]) << ", "
                          << std::hex
                          << ((std::get<8>(memory_accesses[i]) < 16)?("0x0"):("0x")) << std::get<8>(memory_accesses[i]) << ", "
                          << ((std::get<9>(memory_accesses[i]) < 16)?("0x0"):("0x")) << std::get<9>(memory_accesses[i]) << ", "
                          << std::dec
                          << ")\n";
            #endif
            #ifdef PRINT_VALUES
            std::cout << "[" << std::endl;
            for (int i = 0; i < k_shape.size[0]; i++){
                std::cout << "\t[";
                for (int j = 0; j < k_shape.size[1] / 2; j++)
                    std::cout << ((int)temp[i * (k_shape.size[1] / 2) + j]) << ", ";
                std::cout << "]," << std::endl;
            }
            std::cout << "]";
            std::cout << std::endl;
            #endif
            doLowPrecisionWeightPack(temp, output, k_shape.size[0], k_shape.size[1] / 2);
            #ifdef PRINT_VALUES
            std::cout << "[" << std::endl;
            for (int i = 0; i < k_shape.size[0]; i++){
                std::cout << "\t[";
                for (int j = 0; j < k_shape.size[1] / 2; j++)
                    std::cout << ((int)output[i * (k_shape.size[1] / 2) + j]) << ", ";
                std::cout << "]," << std::endl;
            }
            std::cout << "]";
            std::cout << std::endl;
            #endif
            LowPrecision::deallocate(temp);
            return Status::Success;
        }
        Status MultiplyInt8Int4(
            const int8_t* input, Shape input_shape,
            const int8_t* kernel, Shape kernel_shape,
            int32_t* output, Shape output_shape
        ){
            int lhs_columns = input_shape.size[0] ,
                rhs_rows = kernel_shape.size[0] ,
                rhs_columns = kernel_shape.size[1];
            
            if (lhs_columns != rhs_columns)
                return Status::SizesMisMatch;
            int8_t*  rhs = const_cast<int8_t*>(kernel);
            const int8_t*  lhs = input;

            int i;
            for (i = 0 ; (i+4) < rhs_rows ; i+=4){
                LowPrecision::FullyConnected::do4BitMultiplication(
                    lhs, rhs, &output[i], lhs_columns);
                rhs += 4 * (lhs_columns / 2);
            }
            if (rhs_rows - i == 1){
                LowPrecision::FullyConnected::do4BitMultiplication(
                    lhs, rhs, &output[i], lhs_columns);
            }
            else if (rhs_rows - i == 2){
                LowPrecision::FullyConnected::do4BitMultiplication(
                    lhs, rhs, &output[i], lhs_columns);
            }
            else if (rhs_rows - i == 3){
                LowPrecision::FullyConnected::do4BitMultiplication(
                    lhs, rhs, &output[i], lhs_columns);
            }
            else if (rhs_rows - i == 4){
                LowPrecision::FullyConnected::do4BitMultiplication(
                    lhs, rhs, &output[i], lhs_columns);
            }
            return Status::Success;
        }
        void do4BitMultiplication(const int8_t* activation, 
                          int8_t* weights, 
                          int32_t* dst, int size){
            const int8_t* _activation = activation;
            int i, end;
            int8_t mask = 0x0F;
            /* Vector assignments:
                * V_r  = v26-28,31,
                * V_rr = v23-25,30,
                * V_M  = v29,
                * V_W  = V0-3,
                * V_A  = V4,
                * V_AP = V5,
                * V_MW = V8,10-12,
                * V_t  = V9,13-15
            */
            asm volatile(
                "dup v29.16b, %w[mask]\n\t"
                "dup v23.8h, wzr\n\t"
                "dup v24.8h, wzr\n\t"
                "dup v25.8h, wzr\n\t"
                "dup v30.8h, wzr\n\t"
                "mov %w[i], wzr\n\t"
                
                "cmp %w[size], #0\n\t"
                "beq 3f\n\t"

                // Start of Outer Loop Over Weights
                "1:\n\t"
                "ld1 {v0.16b},  [%[weights]], #16\n\t"
                "ld1 {v1.16b},  [%[weights]], #16\n\t"
                "ld1 {v2.16b},  [%[weights]], #16\n\t"
                "ld1 {v3.16b},  [%[weights]], #16\n\t"

                "ld1 {v4.16b},  [%[activation]], #16\n\t"
                "ld1 {v5.16b},  [%[activation]], #16\n\t"

                // AND WL, W, M
                "and v8.16b,  v0.16b, v29.16b\n\t"
                "and v10.16b, v1.16b, v29.16b\n\t"
                "and v11.16b, v2.16b, v29.16b\n\t"
                "and v12.16b, v3.16b, v29.16b\n\t"

                // SHL WL, WL, #4
                "shl v8.16b,  v8.16b,  #4\n\t"
                "shl v10.16b, v10.16b, #4\n\t"
                "shl v11.16b, v11.16b, #4\n\t"
                "shl v12.16b, v12.16b, #4\n\t"

                // SSHR WH, W, #4
                "sshr v0.16b, v0.16b, #4\n\t"
                "sshr v1.16b, v1.16b, #4\n\t"
                "sshr v2.16b, v2.16b, #4\n\t"
                "sshr v3.16b, v3.16b, #4\n\t"

                // SSHR WL, WL, #4
                "sshr v8.16b,  v8.16b,  #4\n\t"
                "sshr v10.16b, v10.16b, #4\n\t"
                "sshr v11.16b, v11.16b, #4\n\t"
                "sshr v12.16b, v12.16b, #4\n\t"

                // SMULL T1.8h, WL.8b, A1.8b
                "smull v13.8h, v8.8b,  v4.8b\n\t"
                "smull v14.8h, v10.8b, v4.8b\n\t"
                "smull v15.8h, v11.8b, v4.8b\n\t"
                "smull v16.8h, v12.8b, v4.8b\n\t"

                // SMLAL2 T1.8h, WL.16b, A1.16b
                "smlal2 v13.8h, v8.16b,  v4.16b\n\t"
                "smlal2 v14.8h, v10.16b, v4.16b\n\t"
                "smlal2 v15.8h, v11.16b, v4.16b\n\t"
                "smlal2 v16.8h, v12.16b, v4.16b\n\t"

                // SMLAL T1.8h, WH.8b, A2.8b
                "smlal v13.8h, v0.8b, v5.8b\n\t"
                "smlal v14.8h, v1.8b, v5.8b\n\t"
                "smlal v15.8h, v2.8b, v5.8b\n\t"
                "smlal v16.8h, v3.8b, v5.8b\n\t"

                // SMLAL2 T1.8h, WH.16b, A2.16b
                "smlal2 v13.8h, v0.16b, v5.16b\n\t"
                "smlal2 v14.8h, v1.16b, v5.16b\n\t"
                "smlal2 v15.8h, v2.16b, v5.16b\n\t"
                "smlal2 v16.8h, v3.16b, v5.16b\n\t"

                // ACCUMULATE ACC, T1
                "sadalp v23.4s, v13.8h\n\t"
                "sadalp v24.4s, v14.8h\n\t"
                "sadalp v25.4s, v15.8h\n\t"
                "sadalp v30.4s, v16.8h\n\t"

                "add %w[i], %w[i], #32\n\t"
                "cmp %w[i], %w[size]\n\t"
                "b.lt 1b\n\t"

                "addv s23, v23.4s\n\t"
                "addv s24, v24.4s\n\t"
                "addv s25, v25.4s\n\t"
                "addv s30, v30.4s\n\t"

                "mov v23.s[1], v24.s[0]\n\t"
                "mov v23.s[2], v25.s[0]\n\t"
                "mov v23.s[3], v30.s[0]\n\t"

                // "ld1 {v30.4s},  [%[dst]]\n\t"
                // "add v30.4s, v30.4s, v23.4s\n\t"
                // "st1 {v30.4s},  [%[dst]]\n\t"
                "st1 {v23.4s},  [%[dst]]\n\t"

                "sub %[activation], %[activation], %[size]\n\t"
                "sub %[weights], %[weights], %[size]\n\t"
                "sub %[weights], %[weights], %[size]\n\t"

                "3:\n\t"

                : [ dst ] "+r"(dst), [ i ] "+r"(i), [ end ] "+r"(end)
                : [ activation ] "r"(_activation), [ weights ] "r"(weights),
                  [ size ] "r"(size), [ mask ] "r"(mask)
                : "v0",  "v1",  "v2",  "v3",
                  "v4",  "v5",  "v6",  "v7",
                  "v8",  "v9",  "v10", "v11",
                  "v12", "v13", "v14", "v15",
                  "v16", "v17", "v18", "v19",
                  "v20", "v21", "v22", "v23",
                  "v24", "v25", "v26", "v27",
                  "v28", "v29", "v30", "v31",
                  "w3",  "w4",  "w5",  "w6"
            );
        }
        uint8_t quantizeTo4BitIntAndPackBitsStep(const int8_t& input, int shift_amount){
            if (input < 0)
                if (input < -8)
                    return (get_as<uint8_t>(-8) << (shift_amount * 4)) & (0x0f << ((shift_amount) * 4));
                else
                    return (get_as<uint8_t>(input) << (shift_amount * 4)) & (0x0f << ((shift_amount) * 4));
            else
                if(input >  7)
                    return (get_as<uint8_t>(7) << (shift_amount * 4)) & (0x0f << ((shift_amount) * 4));
                else
                    return (get_as<uint8_t>(input) << (shift_amount * 4)) & (0x0f << ((shift_amount) * 4));
        }
        
    }
}
#endif
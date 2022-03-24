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
            else if (retval == std::string("F32-Binary"))
                return Method::kFloat32Binary;
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
            DataType input_type, DataType filter_type, 
            DataType output_type, bool Is_FC = false){
            if (!Is_FC)
                return false;
            bool multibatched_enabled = GetVariableFromEnv("LowPrecisionMultiBatched") == "TRUE";
            bool is_multibatched = input_shape.size[0] > 1;
            // Checking for Not-Supported Input DataTypes
            if (
                (input_type != DataType::Int8 && input_type != DataType::Float32 && input_type != DataType::Int32) ||
                filter_type != DataType::Int8 ||
                (output_type != DataType::Float32 && output_type != DataType::Int32))
                return false;
            // checking for the conditions of rejection of multi-batched input
            if (
                (input_shape.number_dims == 2 && 
                input_shape.size[0] % 4 &&
                input_shape.size[0] > 1) ||
                (!multibatched_enabled && is_multibatched)
            )
                return false;

            // if (input_shape.size[0] != 1)
            //     return false;

            // Checking conditions of input shape of any method
            if (method == Method::kInt8Binary ||
                method == Method::kFloat32Binary ||
                method == Method::kFloat16Binary)
                return (!(input_shape.size[input_shape.number_dims - 1] % 128) && !is_multibatched);
            if (method == Method::kInt8Ternary ||
                method == Method::kFloat32Ternary ||
                method == Method::kFloat16Ternary ||
                method == Method::kInt8QuaTernary)
                return (!(input_shape.size[input_shape.number_dims - 1] % 64) && !is_multibatched);
            if (method == Method::kInt8Int4)
                return !(input_shape.size[input_shape.number_dims - 1] % 32);
            // If none of the aboves
            return false;
        }
        
        namespace Int4 {
            void TransformFilterShape(int* shape, int n_dims){
                shape[n_dims - 1] /= 2;
            }
            Status QuantizeFilter(const int8_t* input, Shape k_shape, int8_t* output, MemLayout layout){
                if (k_shape.number_dims != 2)
                    return Status::DimensionsMisMatch;
                if (k_shape.size[1] % 32)
                    return Status::SizesMisMatch; 
                if (GetVariableFromEnv("DismissFilterQuantization") == std::string("TRUE")){
                    doLowPrecisionWeightPack(const_cast<int8_t*>(input), output, k_shape.size[0], k_shape.size[1] / 2);
                }
                else {
                    // std::cerr << "Filter Qantizer Step #1 with shape of " << get_shape_string(k_shape) << std::endl; 
                    int new_weights_length = (k_shape.size[1] / 2) * k_shape.size[0];
                    int8_t* temp = LowPrecision::allocate<int8_t>(new_weights_length);
                    zero_vector(temp, new_weights_length);
                    uint8_t* temp_u = get_pointer_as<uint8_t>(temp);
                    // std::cerr << "Filter Qantizer Step #2\n";
                    #ifdef PRINT_VALUES_DETAILED
                    std::vector<std::tuple<int, int, int, int, int, int, int, int, int, int>> memory_accesses;
                    std::cout << get_shape_string(k_shape) << std::endl;
                    #endif
                    // std::cerr << "Filter Qantizer Step #3\n";
                    if (layout == MemLayout::kColumnMajor)
                        for (int i = 0 ; i < k_shape.size[0] ; i++){
                            for (int j = 0 ; j < k_shape.size[1] ; j++){
                                int cluster_id = j / 32,
                                    container_id = (j % 32) % 16,
                                    shift_amount = ((j % 32) / 16);
                                temp_u[i * (k_shape.size[1]/2) + ((cluster_id * 16) + container_id)] |=
                                    quantizeAndPackBitsStep(input[j * k_shape.size[0] + i], shift_amount);
                                #ifdef PRINT_VALUES_DETAILED
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
                                    quantizeAndPackBitsStep(input[i * k_shape.size[1] + j], shift_amount);
                                #ifdef PRINT_VALUES_DETAILED
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
                    #ifdef PRINT_VALUES_DETAILED
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
                    // std::cerr << "Filter Qantizer Step #4 with shape of (" << k_shape.size[0] << ", " << k_shape.size[1] / 2 << ")" << std::endl;
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
                    // std::cerr << "Filter Qantizer Step #5\n";
                    LowPrecision::deallocate(temp);
                }
                return Status::Success;
            }
            Status MultiplyInt8(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape
            ){
                // In case of multi-batched input
                // std::cout << "Inside MultiplyInt8 with multibatch set to " 
                //           << ((input_shape.number_dims == 2 && input_shape.size[0] != 1)?("ON"):("OFF"))
                //           << std::endl;
                if (input_shape.number_dims == 2 && input_shape.size[0] != 1)
                    return MultiplyInt8MultiBatched(input, input_shape, kernel, kernel_shape, output, output_shape);
                
                int lhs_columns = input_shape.size[0] ,
                    rhs_rows = kernel_shape.size[0] ,
                    rhs_columns = kernel_shape.size[1];
                if (lhs_columns != rhs_columns)
                    return Status::SizesMisMatch;
                int8_t*  rhs = const_cast<int8_t*>(kernel);
                const int8_t*  lhs = input;

                int i;
                for (i = 0 ; (i+4) < rhs_rows ; i+=4){
                    LowPrecision::FullyConnected::Int4::doMultiplication1Col(
                        lhs, rhs, &output[i], lhs_columns);
                    rhs += 4 * (lhs_columns / 2);
                }
                if (rhs_rows - i == 1){
                    LowPrecision::FullyConnected::Int4::doMultiplication1Col(
                        lhs, rhs, &output[i], lhs_columns);
                }
                else if (rhs_rows - i == 2){
                    LowPrecision::FullyConnected::Int4::doMultiplication1Col(
                        lhs, rhs, &output[i], lhs_columns);
                }
                else if (rhs_rows - i == 3){
                    LowPrecision::FullyConnected::Int4::doMultiplication1Col(
                        lhs, rhs, &output[i], lhs_columns);
                }
                else if (rhs_rows - i == 4){
                    LowPrecision::FullyConnected::Int4::doMultiplication1Col(
                        lhs, rhs, &output[i], lhs_columns);
                }
                return Status::Success;
            }
            Status MultiplyInt8MultiBatched(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape
            ){
                int lhs_batches = input_shape.size[0] ,
                    lhs_columns = input_shape.size[1] ,
                    rhs_rows = kernel_shape.size[0] ,
                    rhs_columns = kernel_shape.size[1];
                
                if (lhs_columns != rhs_columns)
                    return Status::SizesMisMatch;
                if (lhs_batches % 4)
                    return Status::NotSupported;

                int8_t* rhs    = const_cast<int8_t*>(kernel);
                int8_t* lhs    = const_cast<int8_t*>(input);
                int32_t* dst_0 = output + 0 * rhs_rows;
                int32_t* dst_1 = output + 1 * rhs_rows;
                int32_t* dst_2 = output + 2 * rhs_rows;
                int32_t* dst_3 = output + 3 * rhs_rows;

                for (int j = 0; j < lhs_batches; j+=4){
                    int i;
                    for (i = 0 ; (i+4) < rhs_rows ; i+=4){
                        LowPrecision::FullyConnected::Int4::doMultiplication(
                            lhs, rhs, 
                            dst_0, dst_1,
                            dst_2, dst_3,
                            lhs_columns);
                        dst_0 += 4;
                        dst_1 += 4;
                        dst_2 += 4;
                        dst_3 += 4;
                        rhs += 4 * (lhs_columns / 2);
                    }
                    if (rhs_rows - i == 1){
                        LowPrecision::FullyConnected::Int4::doMultiplication(
                            lhs, rhs, 
                            dst_0, dst_1,
                            dst_2, dst_3,
                            lhs_columns);
                    }
                    else if (rhs_rows - i == 2){
                        LowPrecision::FullyConnected::Int4::doMultiplication(
                            lhs, rhs, 
                            dst_0, dst_1,
                            dst_2, dst_3,
                            lhs_columns);
                    }
                    else if (rhs_rows - i == 3){
                        LowPrecision::FullyConnected::Int4::doMultiplication(
                            lhs, rhs, 
                            dst_0, dst_1,
                            dst_2, dst_3,
                            lhs_columns);
                    }
                    else if (rhs_rows - i == 4){
                        LowPrecision::FullyConnected::Int4::doMultiplication(
                            lhs, rhs, 
                            dst_0, dst_1,
                            dst_2, dst_3,
                            lhs_columns);
                    }
                    lhs   += 4 * lhs_columns;
                    dst_0 += 4 * rhs_rows;
                    dst_1 += 4 * rhs_rows;
                    dst_2 += 4 * rhs_rows;
                    dst_3 += 4 * rhs_rows;
                }
                return Status::Success;
            }
            void doMultiplication1Col(const int8_t* activation, 
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
            void doMultiplication(const int8_t* activation,
                            int8_t* weights,
                            int32_t* dst_1, int32_t* dst_2,
                            int32_t* dst_3, int32_t* dst_4,
                            int size){
                const int8_t* _activation = activation;
                int i, end;
                int8_t mask = 0x0F;
                /* Vector assignments:
                    * W    -> v0-3
                    * A    -> v4-7
                    * CW   -> v8-11
                    * T    -> v12-15
                    * ACC1 -> v16-19
                    * ACC2 -> v20-23
                    * ACC3 -> v24-27
                    * ACC4 -> v28-31
                */
                asm volatile(
                    #define SET_ZERO(reg) "movi " #reg ".4s, #0\n\t"

                    #define MULL(dst, src1, src2) "smull " #dst ".8h, " #src1 ".8b, " #src2 ".8b\n\t" \
                                                "smlal2 " #dst ".8h, " #src1 ".16b, " #src2 ".16b\n\t"
                    
                    #define ACC(acc, src) "sadalp " #acc ".4s, " #src ".8h\n\t"

                    // Clear accumulators.
                    SET_ZERO(v16)
                    SET_ZERO(v17)
                    SET_ZERO(v18)
                    SET_ZERO(v19)
                    SET_ZERO(v20)
                    SET_ZERO(v21)
                    SET_ZERO(v22)
                    SET_ZERO(v23)
                    SET_ZERO(v24)
                    SET_ZERO(v25)
                    SET_ZERO(v26)
                    SET_ZERO(v27)
                    SET_ZERO(v28)
                    SET_ZERO(v29)
                    SET_ZERO(v30)
                    SET_ZERO(v31)

                    // "dup v29.16b, %w[mask]\n\t"

                    "mov %w[i], wzr\n\t"
                    
                    "cmp %w[size], #0\n\t"
                    "beq 3f\n\t"

                    // Start of Outer Loop Over Weights
                    "1:\n\t"

                    // Create Mask Vector
                    "dup v12.16b, %w[mask]\n\t"

                    // LD W1-4,x-y, WP
                    "ld1 {v0.16b},  [%[weights]], #16\n\t"
                    "ld1 {v1.16b},  [%[weights]], #16\n\t"
                    "ld1 {v2.16b},  [%[weights]], #16\n\t"
                    "ld1 {v3.16b},  [%[weights]], #16\n\t"

                    // LD A1-4,x  , AP
                    "ld1 {v4.16b},  [%[activation]], #16\n\t"
                    "ld1 {v5.16b},  [%[activation]], #16\n\t"
                    "ld1 {v6.16b},  [%[activation]], #16\n\t"
                    "ld1 {v7.16b},  [%[activation]], #16\n\t"

                    // AND CW, W, M
                    "and v8.16b,  v0.16b, v12.16b\n\t"
                    "and v9.16b,  v1.16b, v12.16b\n\t"
                    "and v10.16b, v2.16b, v12.16b\n\t"
                    "and v11.16b, v3.16b, v12.16b\n\t"

                    // SHL CW, CW, #4
                    "shl v8.16b,  v8.16b,  #4\n\t"
                    "shl v9.16b,  v9.16b,  #4\n\t"
                    "shl v10.16b, v10.16b, #4\n\t"
                    "shl v11.16b, v11.16b, #4\n\t"

                    // SSHR CW, CW, #4
                    "sshr v8.16b,  v8.16b,  #4\n\t"
                    "sshr v9.16b,  v9.16b,  #4\n\t"
                    "sshr v10.16b, v10.16b, #4\n\t"
                    "sshr v11.16b, v11.16b, #4\n\t"

                    // M T, CW, A1,x
                    MULL(v12, v8,  v4)
                    MULL(v13, v9,  v4)
                    MULL(v14, v10, v4)
                    MULL(v15, v11, v4)

                    // ACC ACC1, T
                    ACC(v16, v12)
                    ACC(v17, v13)
                    ACC(v18, v14)
                    ACC(v19, v15)

                    // M T, CW, A2,x
                    MULL(v12, v8,  v5)
                    MULL(v13, v9,  v5)
                    MULL(v14, v10, v5)
                    MULL(v15, v11, v5)

                    // ACC ACC2, T
                    ACC(v20, v12)
                    ACC(v21, v13)
                    ACC(v22, v14)
                    ACC(v23, v15)

                    // M T, CW, A3,x
                    MULL(v12, v8,  v6)
                    MULL(v13, v9,  v6)
                    MULL(v14, v10, v6)
                    MULL(v15, v11, v6)

                    // ACC ACC3, T
                    ACC(v24, v12)
                    ACC(v25, v13)
                    ACC(v26, v14)
                    ACC(v27, v15)

                    // M T, CW, A4,x
                    MULL(v12, v8,  v7)
                    MULL(v13, v9,  v7)
                    MULL(v14, v10, v7)
                    MULL(v15, v11, v7)

                    // ACC ACC4, T
                    ACC(v28, v12)
                    ACC(v29, v13)
                    ACC(v30, v14)
                    ACC(v31, v15)

                    // LD A1-4,y  , AP
                    "ld1 {v4.16b},  [%[activation]], #16\n\t"
                    "ld1 {v5.16b},  [%[activation]], #16\n\t"
                    "ld1 {v6.16b},  [%[activation]], #16\n\t"
                    "ld1 {v7.16b},  [%[activation]], #16\n\t"

                    // SSHR CW, W, #4
                    "sshr v8.16b,  v0.16b, #4\n\t"
                    "sshr v9.16b,  v1.16b, #4\n\t"
                    "sshr v10.16b, v2.16b, #4\n\t"
                    "sshr v11.16b, v3.16b, #4\n\t"

                    // M T, CW, A1,x
                    MULL(v12, v8,  v4)
                    MULL(v13, v9,  v4)
                    MULL(v14, v10, v4)
                    MULL(v15, v11, v4)

                    // ACC ACC1, T
                    ACC(v16, v12)
                    ACC(v17, v13)
                    ACC(v18, v14)
                    ACC(v19, v15)

                    // M T, CW, A2,x
                    MULL(v12, v8,  v5)
                    MULL(v13, v9,  v5)
                    MULL(v14, v10, v5)
                    MULL(v15, v11, v5)

                    // ACC ACC2, T
                    ACC(v20, v12)
                    ACC(v21, v13)
                    ACC(v22, v14)
                    ACC(v23, v15)

                    // M T, CW, A3,x
                    MULL(v12, v8,  v6)
                    MULL(v13, v9,  v6)
                    MULL(v14, v10, v6)
                    MULL(v15, v11, v6)

                    // ACC ACC3, T
                    ACC(v24, v12)
                    ACC(v25, v13)
                    ACC(v26, v14)
                    ACC(v27, v15)

                    // M T, CW, A4,x
                    MULL(v12, v8,  v7)
                    MULL(v13, v9,  v7)
                    MULL(v14, v10, v7)
                    MULL(v15, v11, v7)

                    // ACC ACC4, T
                    ACC(v28, v12)
                    ACC(v29, v13)
                    ACC(v30, v14)
                    ACC(v31, v15)

                    "add %w[i], %w[i], #32\n\t"
                    "cmp %w[i], %w[size]\n\t"
                    "b.lt 1b\n\t"

                    // Copied From Ruy::Kernel8bitNeon
                    // Reduce 32bit accumulators horizontally.
                    "addp v16.4s, v16.4s, v17.4s\n"
                    "addp v18.4s, v18.4s, v19.4s\n"
                    "addp v20.4s, v20.4s, v21.4s\n"
                    "addp v22.4s, v22.4s, v23.4s\n"
                    "addp v24.4s, v24.4s, v25.4s\n"
                    "addp v26.4s, v26.4s, v27.4s\n"
                    "addp v28.4s, v28.4s, v29.4s\n"
                    "addp v30.4s, v30.4s, v31.4s\n"

                    // Reduce 32bit accumulators horizontally, second pass
                    // (each pass adds pairwise. we need to add 4-wise).
                    "addp v16.4s, v16.4s, v18.4s\n"
                    "addp v17.4s, v20.4s, v22.4s\n"
                    "addp v18.4s, v24.4s, v26.4s\n"
                    "addp v19.4s, v28.4s, v30.4s\n"

                    "st1 {v16.4s},  [%[dst_1]], #16\n\t"
                    "st1 {v17.4s},  [%[dst_2]], #16\n\t"
                    "st1 {v18.4s},  [%[dst_3]], #16\n\t"
                    "st1 {v19.4s},  [%[dst_4]], #16\n\t"

                    "sub %[activation], %[activation], %[size]\n\t"
                    "sub %[activation], %[activation], %[size]\n\t"
                    "sub %[activation], %[activation], %[size]\n\t"
                    "sub %[activation], %[activation], %[size]\n\t"
                    "sub %[weights], %[weights], %[size]\n\t"
                    "sub %[weights], %[weights], %[size]\n\t"

                    "3:\n\t"

                    : [ dst_1 ] "+r"(dst_1), [ dst_2 ] "+r"(dst_2),
                    [ dst_3 ] "+r"(dst_3), [ dst_4 ] "+r"(dst_4),
                    [ i ] "+r"(i), [ end ] "+r"(end)
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
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount){
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
        namespace Binary{
            void TransformFilterShape(int* shape, int n_dims){
                shape[n_dims - 1] /= 8;
            }
            Status QuantizeFilter(const int8_t* input, Shape k_shape, int8_t* output, MemLayout layout){
                if (k_shape.number_dims != 2)
                    return Status::DimensionsMisMatch;
                if (k_shape.size[1] % 128)
                    return Status::SizesMisMatch; 
                if (GetVariableFromEnv("DismissFilterQuantization") == std::string("TRUE")){
                    doLowPrecisionWeightPack(const_cast<int8_t*>(input), output, k_shape.size[0], k_shape.size[1] / 8);
                }
                else {
                    // std::cerr << "Filter Qantizer Step #1 with shape of " << get_shape_string(k_shape) << std::endl;
                    int new_weights_length = (k_shape.size[1] / 8) * k_shape.size[0];
                    int8_t* temp = LowPrecision::allocate<int8_t>(new_weights_length);
                    zero_vector(temp, new_weights_length);
                    uint8_t* temp_u = get_pointer_as<uint8_t>(temp);
                    // std::cerr << "Filter Qantizer Step #2\n";
                    #ifdef PRINT_VALUES_DETAILED
                    std::vector<std::tuple<int, int, int, int, int, int, int, int, int, int, int, int>> memory_accesses;
                    std::cout << get_shape_string(k_shape) << std::endl;
                    #endif
                    // std::cerr << "Filter Qantizer Step #3\n";
                    if (layout == MemLayout::kColumnMajor)
                        for (int i = 0 ; i < k_shape.size[0] ; i++){
                            for (int j = 0 ; j < k_shape.size[1] ; j++){
                                int cluster_id = j / 128,
                                    container_id = (j % 128) % 16,
                                    shift_amount = (8 - 1) - ((j % 128) / 16);
                                temp_u[i * (k_shape.size[1]/8) + ((cluster_id * 16) + container_id)] |=
                                    quantizeAndPackBitsStep(input[j * k_shape.size[0] + i], shift_amount);
                                #ifdef PRINT_VALUES_DETAILED
                                std::tuple<int, int, int, int, int, int, int, int, int, int, int, int> access;
                                std::get<0>(access) = i;
                                std::get<1>(access) = j;
                                std::get<2>(access) = i * (k_shape.size[1]/8) + ((cluster_id * 16) + container_id);
                                std::get<3>(access) = shift_amount;
                                std::get<4>(access) = cluster_id;
                                std::get<5>(access) = container_id;
                                std::get<6>(access) = j * k_shape.size[0] + i;
                                std::get<7>(access) = input[j * k_shape.size[0] + i];
                                std::get<8>(access) = 0x01 << (shift_amount);
                                std::get<9>(access) = (input[j * k_shape.size[0] + i] << shift_amount) & (0x01 << shift_amount);
                                std::get<10>(access) = quantizeAndPackBitsStep(input[j * k_shape.size[0] + i], shift_amount);
                                std::get<11>(access) = temp_u[i * (k_shape.size[1]/8) + ((cluster_id * 16) + container_id)];
                                memory_accesses.push_back(access);
                                #endif
                            }
                        }
                    else
                        for (int i = 0 ; i < k_shape.size[0] ; i++){
                            for (int j = 0 ; j < k_shape.size[1] ; j++){
                                int cluster_id = j / 128,
                                    container_id = (j % 128) % 16,
                                    shift_amount = (8 - 1) - ((j % 128) / 16);
                                temp_u[i * (k_shape.size[1]/8) + ((cluster_id * 16) + container_id)] |=
                                    quantizeAndPackBitsStep(input[i * k_shape.size[1] + j], shift_amount);
                                #ifdef PRINT_VALUES_DETAILED
                                std::tuple<int, int, int, int, int, int, int, int, int, int, int, int> access;
                                std::get<0>(access) = i;
                                std::get<1>(access) = j;
                                std::get<2>(access) = i * (k_shape.size[1]/8) + ((cluster_id * 16) + container_id);
                                std::get<3>(access) = shift_amount;
                                std::get<4>(access) = cluster_id;
                                std::get<5>(access) = container_id;
                                std::get<6>(access) = i * k_shape.size[1] + j;
                                std::get<7>(access) = input[i * k_shape.size[1] + j];
                                std::get<8>(access) = 0x01 << shift_amount;
                                std::get<9>(access) = (input[i * k_shape.size[1] + j] << shift_amount) & (0x01 << shift_amount);
                                std::get<10>(access) = quantizeAndPackBitsStep(input[i * k_shape.size[1] + j], shift_amount);
                                std::get<11>(access) = temp_u[i * (k_shape.size[1]/8) + ((cluster_id * 16) + container_id)];
                                memory_accesses.push_back(access);
                                #endif
                            }
                        }
                    #ifdef PRINT_VALUES_DETAILED
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
                                << ((std::get<10>(memory_accesses[i]) < 16)?("0x0"):("0x")) << std::get<10>(memory_accesses[i]) << ", "
                                << ((std::get<11>(memory_accesses[i]) < 16)?("0x0"):("0x")) << std::get<11>(memory_accesses[i]) << ", "
                                << std::dec
                                << ")\n";
                    #endif
                    #ifdef PRINT_VALUES
                    std::cout << "[" << std::endl;
                    for (int i = 0; i < k_shape.size[0]; i++){
                        std::cout << "\t[";
                        for (int j = 0; j < k_shape.size[1] / 8; j++)
                            std::cout << ((int)temp[i * (k_shape.size[1] / 8) + j]) << ", ";
                        std::cout << "]," << std::endl;
                    }
                    std::cout << "]";
                    std::cout << std::endl;
                    #endif
                    // std::cerr << "Filter Qantizer Step #4 with shape of (" << k_shape.size[0] << ", " << k_shape.size[1] / 8 << ")" << std::endl;
                    doLowPrecisionWeightPack(temp, output, k_shape.size[0], k_shape.size[1] / 8);
                    #ifdef PRINT_VALUES
                    std::cout << "After Packing:" << std::endl;
                    std::cout << "[" << std::endl;
                    for (int i = 0; i < k_shape.size[0]; i++){
                        std::cout << "\t[";
                        for (int j = 0; j < k_shape.size[1] / 8; j++)
                            std::cout << ((int)output[i * (k_shape.size[1] / 8) + j]) << ", ";
                        std::cout << "]," << std::endl;
                    }
                    std::cout << "]";
                    std::cout << std::endl;
                    #endif
                    // std::cerr << "Filter Qantizer Step #5\n";
                    LowPrecision::deallocate(temp);
                }
                return Status::Success;
            }
            Status MultiplyInt8SingleBatch(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape
            ){
                int lhs_columns = input_shape.size[0] ,
                    rhs_rows = kernel_shape.size[0] ,
                    rhs_columns = kernel_shape.size[1];
                
                if (lhs_columns != rhs_columns)
                    return Status::SizesMisMatch;
                if(lhs_columns == 0 || rhs_rows == 0)
                    return Status::Success;
                
                int8_t*         _kernel = const_cast<int8_t*>(kernel);
                const int8_t*   _input  = input;
                int32_t*        _output = output;
                int i, j, end;

                asm volatile(
                    "mov %w[j], wzr\n\t"

                    // "cmp %w[rows], #0\n\t"
                    // "beq 5f\n\t"

                    "0:\n\t"
                    "mov %w[i], wzr\n\t"
                    "dup v23.4s, wzr\n\t"
                    "dup v24.4s, wzr\n\t"
                    "dup v25.4s, wzr\n\t"
                    "dup v30.4s, wzr\n\t"
                    
                    // "cmp %w[size], #0\n\t"
                    // "beq 3f\n\t"

                    // Start of Outer Loop Over Weights
                    "1:\n\t"
                    "ld1 {v0.16b},  [%[weights]], #16\n\t"
                    "ld1 {v1.16b},  [%[weights]], #16\n\t"
                    "ld1 {v2.16b},  [%[weights]], #16\n\t"
                    "ld1 {v3.16b},  [%[weights]], #16\n\t"

                    "ld1 {v4.16b},  [%[activation]], #16\n\t"
                    // Generate negate of activations
                    "sqneg v5.16b,  v4.16b\n\t"

                    "add %w[end], %w[i], #128\n\t"
                    "tst %w[size], %w[end]\n\t"
                    "csel %w[end], %w[end], %w[size], lo\n\t"

                    "dup v26.8h, wzr\n\t"
                    "dup v27.8h, wzr\n\t"
                    "dup v28.8h, wzr\n\t"
                    "dup v31.8h, wzr\n\t"
                    
                    // Start of Inner Loop Over Activations
                    "2:\n\t"

                    // SSHR T1, W, #7
                    "sshr v8.16b,  v0.16b, #7\n\t"
                    "sshr v10.16b, v1.16b, #7\n\t"
                    "sshr v11.16b, v2.16b, #7\n\t"
                    "sshr v12.16b, v3.16b, #7\n\t"
                    
                    // AND APt, T1, AP
                    "and v9.16b,  v8.16b,  v5.16b\n\t"
                    "and v13.16b, v10.16b, v5.16b\n\t"
                    "and v14.16b, v11.16b, v5.16b\n\t"
                    "and v15.16b, v12.16b, v5.16b\n\t"

                    // NOT T1, T1
                    "not v8.16b,  v8.16b\n\t"
                    "not v10.16b, v10.16b\n\t"
                    "not v11.16b, v11.16b\n\t"
                    "not v12.16b, v12.16b\n\t"

                    // AND At, T1, A
                    "and v8.16b,  v8.16b,  v4.16b\n\t"
                    "and v10.16b, v10.16b, v4.16b\n\t"
                    "and v11.16b, v11.16b, v4.16b\n\t"
                    "and v12.16b, v12.16b, v4.16b\n\t"

                    "ld1 {v4.16b},  [%[activation]], #16\n\t"

                    // ORR AT, At, APt
                    "orr v9.16b,  v8.16b,  v9.16b\n\t"
                    "orr v13.16b, v10.16b, v13.16b\n\t"
                    "orr v14.16b, v11.16b, v14.16b\n\t"
                    "orr v15.16b, v12.16b, v15.16b\n\t"

                    // SHL W, W, 7
                    "shl v0.16b, v0.16b, #1\n\t"
                    "shl v1.16b, v1.16b, #1\n\t"
                    "shl v2.16b, v2.16b, #1\n\t"
                    "shl v3.16b, v3.16b, #1\n\t"
                    
                    // SADALP MiniAC, AT
                    "sadalp v26.8h, v9.16b\n\t"
                    "sadalp v27.8h, v13.16b\n\t"
                    "sadalp v28.8h, v14.16b\n\t"
                    "sadalp v31.8h, v15.16b\n\t"

                    // Generate negate of activations
                    "sqneg v5.16b,  v4.16b\n\t"

                    "add %w[i], %w[i], #16\n\t"
                    "cmp %w[i], %w[end]\n\t"
                    "b.lt 2b\n\t"

                    "sadalp v23.4s, v26.8h\n\t"
                    "sadalp v24.4s, v27.8h\n\t"
                    "sadalp v25.4s, v28.8h\n\t"
                    "sadalp v30.4s, v31.8h\n\t"

                    "cmp %w[i], %w[size]\n\t"
                    "b.lt 1b\n\t"

                    "addv s23, v23.4s\n\t"
                    "addv s24, v24.4s\n\t"
                    "addv s25, v25.4s\n\t"
                    "addv s30, v30.4s\n\t"

                    "mov v23.s[1], v24.s[0]\n\t"
                    "mov v23.s[2], v25.s[0]\n\t"
                    "mov v23.s[3], v30.s[0]\n\t"

                    "st1 {v23.4s},  [%[dst]], #16\n\t"

                    "mov %[activation], %[lhs_base]\n\t"

                    "3:\n\t"

                    "add %w[j], %w[j], #4\n\t"
                    "cmp %w[j], %w[rows]\n\t"
                    "b.lt 0b\n\t"


                    : [ dst ]        "+r" (_output),     [ i ]           "+r" (i),
                      [ j ]          "+r" (j),           [ end ]         "+r" (end)

                    : [ activation ] "r"  (_input),      [ lhs_base ]    "r"  (input),
                      [ weights ]    "r"  (_kernel),     [ size ]        "r"  (lhs_columns),
                      [ rows ]       "r"  (rhs_rows)

                    : "v0",  "v1",  "v2",  "v3",
                      "v4",  "v5",  "v6",  "v7",
                      "v8",  "v9",  "v10", "v11",
                      "v12", "v13", "v14", "v15",
                      "v16", "v17", "v18", "v19",
                      "v20", "v21", "v22", "v23",
                      "v24", "v25", "v26", "v27",
                      "v28", "v29", "v30", "v31"
                );

                return Status::Success;
            }
            Status MultiplyInt8(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape
            ){
                // In case of multi-batched input
                // std::cout << "Inside MultiplyInt8Binary with multibatch set to " 
                //           << ((input_shape.number_dims == 2 && input_shape.size[0] != 1)?("ON"):("OFF"))
                //           << std::endl;
                if (input_shape.number_dims == 2 && input_shape.size[0] != 1)
                    return MultiplyInt8MultiBatched(input, input_shape, kernel, kernel_shape, output, output_shape);
                
                int lhs_columns = input_shape.size[0] ,
                    rhs_rows = kernel_shape.size[0] ,
                    rhs_columns = kernel_shape.size[1];
                
                if (lhs_columns != rhs_columns)
                    return Status::SizesMisMatch;
                int8_t*  rhs = const_cast<int8_t*>(kernel);
                const int8_t*  lhs = input;

                // std::cout << "MultiplyInt8Binary ";
                // std::cout << lhs_columns << ", ";
                // std::cout << rhs_rows << ", ";
                // std::cout << rhs_columns << ", ";
                // std::cout << std::endl;

                int i;
                for (i = 0 ; (i+4) < rhs_rows ; i+=4){
                    LowPrecision::FullyConnected::Binary::doMultiplication1Col(
                        lhs, rhs, &output[i], lhs_columns);
                    rhs += 4 * (lhs_columns / 8);
                }
                if (rhs_rows - i == 1){
                    LowPrecision::FullyConnected::Binary::doMultiplication1Col(
                        lhs, rhs, &output[i], lhs_columns);
                }
                else if (rhs_rows - i == 2){
                    LowPrecision::FullyConnected::Binary::doMultiplication1Col(
                        lhs, rhs, &output[i], lhs_columns);
                }
                else if (rhs_rows - i == 3){
                    LowPrecision::FullyConnected::Binary::doMultiplication1Col(
                        lhs, rhs, &output[i], lhs_columns);
                }
                else if (rhs_rows - i == 4){
                    LowPrecision::FullyConnected::Binary::doMultiplication1Col(
                        lhs, rhs, &output[i], lhs_columns);
                }
                return Status::Success;
            }
            Status MultiplyInt8MultiBatched(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape
            ){
                int lhs_batches = input_shape.size[0] ,
                    lhs_columns = input_shape.size[1] ,
                    rhs_rows = kernel_shape.size[0] ,
                    rhs_columns = kernel_shape.size[1];
                
                if (lhs_columns != rhs_columns)
                    return Status::SizesMisMatch;
                if (lhs_batches % 4)
                    return Status::NotSupported;

                int8_t* rhs    = const_cast<int8_t*>(kernel);
                int8_t* lhs    = const_cast<int8_t*>(input);
                int32_t* dst_0 = output + 0 * rhs_rows;
                int32_t* dst_1 = output + 1 * rhs_rows;
                int32_t* dst_2 = output + 2 * rhs_rows;
                int32_t* dst_3 = output + 3 * rhs_rows;

                for (int j = 0; j < lhs_batches; j+=4){
                    int i;
                    for (i = 0 ; (i+4) < rhs_rows ; i+=4){
                        LowPrecision::FullyConnected::Binary::doMultiplication(
                            lhs, rhs, 
                            dst_0, dst_1,
                            dst_2, dst_3,
                            lhs_columns);
                        dst_0 += 4;
                        dst_1 += 4;
                        dst_2 += 4;
                        dst_3 += 4;
                        rhs += 4 * (lhs_columns / 8);
                    }
                    if (rhs_rows - i == 1){
                        LowPrecision::FullyConnected::Binary::doMultiplication(
                            lhs, rhs, 
                            dst_0, dst_1,
                            dst_2, dst_3,
                            lhs_columns);
                    }
                    else if (rhs_rows - i == 2){
                        LowPrecision::FullyConnected::Binary::doMultiplication(
                            lhs, rhs, 
                            dst_0, dst_1,
                            dst_2, dst_3,
                            lhs_columns);
                    }
                    else if (rhs_rows - i == 3){
                        LowPrecision::FullyConnected::Binary::doMultiplication(
                            lhs, rhs, 
                            dst_0, dst_1,
                            dst_2, dst_3,
                            lhs_columns);
                    }
                    else if (rhs_rows - i == 4){
                        LowPrecision::FullyConnected::Binary::doMultiplication(
                            lhs, rhs, 
                            dst_0, dst_1,
                            dst_2, dst_3,
                            lhs_columns);
                    }
                    lhs   += 4 * lhs_columns;
                    dst_0 += 4 * rhs_rows;
                    dst_1 += 4 * rhs_rows;
                    dst_2 += 4 * rhs_rows;
                    dst_3 += 4 * rhs_rows;
                }
                return Status::Success;
            }
            void doMultiplication1Col(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst, int size){
                const int8_t* _activation = activation;
                int i, end;
                asm volatile(
                    "dup v23.4s, wzr\n\t"
                    "dup v24.4s, wzr\n\t"
                    "dup v25.4s, wzr\n\t"
                    "dup v30.4s, wzr\n\t"
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
                    // Generate negate of activations
                    "sqneg v5.16b,  v4.16b\n\t"

                    "add %w[end], %w[i], #128\n\t"
                    "tst %w[size], %w[end]\n\t"
                    "csel %w[end], %w[end], %w[size], lo\n\t"

                    "dup v26.8h, wzr\n\t"
                    "dup v27.8h, wzr\n\t"
                    "dup v28.8h, wzr\n\t"
                    "dup v31.8h, wzr\n\t"
                    
                    // Start of Inner Loop Over Activations
                    "2:\n\t"

                    // SSHR T1, W, #7
                    "sshr v8.16b,  v0.16b, #7\n\t"
                    "sshr v10.16b, v1.16b, #7\n\t"
                    "sshr v11.16b, v2.16b, #7\n\t"
                    "sshr v12.16b, v3.16b, #7\n\t"
                    
                    // AND APt, T1, AP
                    "and v9.16b,  v8.16b,  v5.16b\n\t"
                    "and v13.16b, v10.16b, v5.16b\n\t"
                    "and v14.16b, v11.16b, v5.16b\n\t"
                    "and v15.16b, v12.16b, v5.16b\n\t"

                    // NEG T1, T1
                    "not v8.16b,  v8.16b\n\t"
                    "not v10.16b, v10.16b\n\t"
                    "not v11.16b, v11.16b\n\t"
                    "not v12.16b, v12.16b\n\t"

                    // AND At, T1, A
                    "and v8.16b,  v8.16b,  v4.16b\n\t"
                    "and v10.16b, v10.16b, v4.16b\n\t"
                    "and v11.16b, v11.16b, v4.16b\n\t"
                    "and v12.16b, v12.16b, v4.16b\n\t"

                    "ld1 {v4.16b},  [%[activation]], #16\n\t"

                    // ORR AT, At, APt
                    "orr v9.16b,  v8.16b,  v9.16b\n\t"
                    "orr v13.16b, v10.16b, v13.16b\n\t"
                    "orr v14.16b, v11.16b, v14.16b\n\t"
                    "orr v15.16b, v12.16b, v15.16b\n\t"

                    "shl v0.16b, v0.16b, #1\n\t"
                    "shl v1.16b, v1.16b, #1\n\t"
                    "shl v2.16b, v2.16b, #1\n\t"
                    "shl v3.16b, v3.16b, #1\n\t"

                    "sadalp v26.8h, v9.16b\n\t"
                    "sadalp v27.8h, v13.16b\n\t"
                    "sadalp v28.8h, v14.16b\n\t"
                    "sadalp v31.8h, v15.16b\n\t"

                    // Generate negate of activations
                    "sqneg v5.16b,  v4.16b\n\t"

                    "add %w[i], %w[i], #16\n\t"
                    "cmp %w[i], %w[end]\n\t"
                    "b.lt 2b\n\t"

                    "sadalp v23.4s, v26.8h\n\t"
                    "sadalp v24.4s, v27.8h\n\t"
                    "sadalp v25.4s, v28.8h\n\t"
                    "sadalp v30.4s, v31.8h\n\t"

                    "cmp %w[i], %w[size]\n\t"
                    "b.lt 1b\n\t"

                    "addv s23, v23.4s\n\t"
                    "addv s24, v24.4s\n\t"
                    "addv s25, v25.4s\n\t"
                    "addv s30, v30.4s\n\t"

                    "mov v23.s[1], v24.s[0]\n\t"
                    "mov v23.s[2], v25.s[0]\n\t"
                    "mov v23.s[3], v30.s[0]\n\t"

                    "st1 {v23.4s},  [%[dst]]\n\t"

                    "sub %[activation], %[activation], %[size]\n\t"
                    "sub %[weights], %[weights], %[size], asr #1\n\t"

                    "3:\n\t"

                    : [ dst ] "+r"(dst), [ i ] "+r"(i), [ end ] "+r"(end)
                    : [ activation ] "r"(_activation), [ weights ] "r"(weights),
                      [ size ] "r"(size)
                    : "v0",  "v1",  "v2",  "v3",
                      "v4",  "v5",  "v6",  "v7",
                      "v8",  "v9",  "v10", "v11",
                      "v12", "v13", "v14", "v15",
                      "v16", "v17", "v18", "v19",
                      "v20", "v21", "v22", "v23",
                      "v24", "v25", "v26", "v27",
                      "v28", "v29", "v30", "v31"
                );
            }
            void doMultiplication(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int32_t* dst_2,
                                    int32_t* dst_3, int32_t* dst_4,
                                    int size){}
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount){
                if (input < 0)
                    return 0x01 << shift_amount;
                else
                    return 0;
            }
        }
        void doScallingFactorMultiplication(int32_t* input, const float* scalling_factor, float* output,
                                            int batch_n, int input_n){
            for(int i = 0 ; i < batch_n ; i++)
                for(int j = 0 ; j < input_n ; j++)
                    output[i * input_n +  j] = input[i * input_n +  j] * scalling_factor[j];
            return;
        }
    }
}
#endif
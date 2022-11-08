#ifdef BAZEL_BUILD
#include "../low_precision_fully_connected.h"
#else
#include "../low_precision_fully_connected.h"
#endif
#ifdef IS_ARM
namespace LowPrecision{
    namespace FullyConnected{
        using ::LowPrecision::Method;
        using ::LowPrecision::Shape;
        using ::LowPrecision::Status;
        using ::LowPrecision::DataType;
        using ::LowPrecision::MemLayout;
        using ::LowPrecision::MulParams;
        namespace Int4InputsInt4Weights {
            // #define W4A4_DECREASE_CONCURRENT_BATCH 1
            // #define W4A4_USE_SINGLE_BATCH_FOR_MULTI 1
            #define W4A4_USE_16_BIT_ACCUMULATOR 1
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape){
                int padding_size = (shape.size[1] % 32)?(32 - (shape.size[1] % 32)):(0);
                Shape new_shape;
                new_shape.number_dims = shape.number_dims;
                new_shape.size = new int[new_shape.number_dims];
                new_shape.size[0] = shape.size[0];
                new_shape.size[1] = shape.size[1] + padding_size;
                new_shape.flatsize = new_shape.size[0] * new_shape.size[1];
                int8_t* new_weight = allocate<int8_t>(new_shape.flatsize);
                for(int i = 0 ; i < shape.size[0] ; i++){
                    for(int j = 0 ; j < shape.size[1] ; j++)
                        new_weight[i * new_shape.size[1] + j] = weight[i * shape.size[1] + j];
                    for (int j = shape.size[1]; j < shape.size[1] + padding_size; j++)
                        new_weight[i * new_shape.size[1] + j] = 0;
                }
                return new_weight;
            }
            size_t TransformFilterShape(int* shape, int n_dims){
                shape[n_dims - 1] = ceil(shape[n_dims - 1] / 16.0) * 16 / (8 / 4);
                return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
            }
            size_t TransformInputShape(int* shape, int n_dims){
                shape[n_dims - 1] = ::ceil(shape[n_dims - 1] / 16.0) * 16 / (8 / 4);
                return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
            }
            Status QuantizeFilter(const int8_t* input, Shape k_shape, int8_t* output, MemLayout layout){
                if (k_shape.number_dims != 2)
                    return Status::DimensionsMisMatch;
                // if (k_shape.size[1] % 32)
                //     return Status::SizesMisMatch;
                if (k_shape.size[0] % 4)
                    return Status::SizesMisMatch;
                if (layout != MemLayout::kRowMajor)
                    return Status::WrongMemLayout;
                if (GetVariableFromEnv("DismissFilterQuantization") == std::string("TRUE")){
                    doLowPrecisionWeightPack(const_cast<int8_t*>(input), output, k_shape.size[0], k_shape.size[1] / 2);
                }
                else {
                    int new_weights_length = k_shape.size[0] * (k_shape.size[1] / 2);
                    int8_t* temp = LowPrecision::allocate<int8_t>(new_weights_length);
                    uint8_t* temp_u = get_pointer_as<uint8_t>(temp);
                    int i , size = k_shape.flatsize;
                    asm volatile(
                        "mov %w[i], wzr\n\t"
                        "movi v31.16b, #15\n\t"
                        "movi v30.16b, #7\n\t"
                        "movi v29.16b, #248\n\t"
                        
                        "cmp %w[size], #0\n\t"
                        "beq 3f\n\t"

                        // Start of Outer Loop Over Weights
                        "1:\n\t"
                        "ld1 {v0.16b},  [%[input]], #16\n\t"
                        "ld1 {v1.16b},  [%[input]], #16\n\t"

                        // Saturate to 7
                        // CMGE AT, A, 7
                        "cmge v2.16b, v0.16b, v30.16b\n\t"
                        "cmge v3.16b, v1.16b, v30.16b\n\t"
                        // NOT AT, AT
                        "not v2.16b, v2.16b\n\t"
                        "not v3.16b, v3.16b\n\t"
                        // AND A, A, AT
                        "and v0.16b, v0.16b, v2.16b\n\t"
                        "and v1.16b, v1.16b, v3.16b\n\t"
                        // NOT ATT, AT
                        "not v2.16b, v2.16b\n\t"
                        "not v3.16b, v3.16b\n\t"
                        // AND ATTT, A, ATT
                        "and v2.16b, v30.16b, v2.16b\n\t"
                        "and v3.16b, v30.16b, v3.16b\n\t"
                        // ORR A, A, ATT
                        "orr v0.16b, v0.16b, v2.16b\n\t"
                        "orr v1.16b, v1.16b, v3.16b\n\t"
                        
                        // Saturate to -8
                        // CMGE AT, A, -8
                        "cmge v2.16b, v0.16b, v29.16b\n\t"
                        "cmge v3.16b, v1.16b, v29.16b\n\t"
                        // AND A, A, AT
                        "and v0.16b, v0.16b, v2.16b\n\t"
                        "and v1.16b, v1.16b, v3.16b\n\t"
                        // NOT ATT, AT
                        "not v2.16b, v2.16b\n\t"
                        "not v3.16b, v3.16b\n\t"
                        // AND ATTT, A, ATT
                        "and v2.16b, v29.16b, v2.16b\n\t"
                        "and v3.16b, v29.16b, v3.16b\n\t"
                        // ORR A, A, ATTT
                        "orr v0.16b, v0.16b, v2.16b\n\t"
                        "orr v1.16b, v1.16b, v3.16b\n\t"

                        // Pack 2 Saturated Int4 in 1 Int8
                        // AND AL, AL, 0x0F
                        "and v0.16b, v0.16b, v31.16b\n\t"
                        // SHL AH, AH, #4
                        "shl v1.16b, v1.16b, #4\n\t"
                        // ORR AM, AL, Ah
                        "orr v0.16b, v0.16b, v1.16b\n\t"

                        // ST1 AM, output
                        "st1 {v0.4s},  [%[output]], #16\n\t"

                        "add %w[i], %w[i], #32\n\t"
                        "cmp %w[i], %w[size]\n\t"
                        "b.lt 1b\n\t"

                        "sub %[input], %[input], %[size]\n\t"

                        "3:\n\t"

                        : [ output ] "+r"(temp_u), [ i ] "+r"(i)
                        : [ input ]  "r" (input), [ size ] "r"(size)
                        : "v0",  "v1",  "v2",  "v3",
                          "v28", "v29", "v30", "v31",
                          "w3",  "w4",  "w5",  "w6"
                    );
                    doLowPrecisionWeightPack(temp, output, k_shape.size[0], k_shape.size[1] / 2);
                    LowPrecision::deallocate(temp);
                }
                return Status::Success;
            }
            Status QuantizeInput(const int8_t* input, Shape shape, int8_t* output, MemLayout layout){
                if (shape.size[shape.number_dims - 1] % 32)
                    return Status::SizesMisMatch; 
                if (layout != MemLayout::kRowMajor)
                    return Status::WrongMemLayout;
                bool is_multibatched = shape.number_dims == 2 && shape.size[0] > 1;
                if (is_multibatched && shape.size[0] % 4)
                    return Status::SizesMisMatch; 
                if (GetVariableFromEnv("DismissInputQuantization") == std::string("TRUE") ||
                    GetVariableFromEnv("DismissQuantization") == std::string("TRUE")){
                    std::copy(input, input + (shape.flatsize / 2), output);
                }
                else {
                    int8_t* temp = output;
                    if (is_multibatched){
                        int new_weights_length = ((int)shape.flatsize / 2);
                        temp = LowPrecision::allocate<int8_t>(new_weights_length);
                    }
                    int i , size = shape.flatsize;
                    asm volatile(
                        "mov %w[i], wzr\n\t"
                        "movi v31.16b, #15\n\t"
                        "movi v30.16b, #7\n\t"
                        "movi v29.16b, #248\n\t"
                        
                        "cmp %w[size], #0\n\t"
                        "beq 3f\n\t"

                        // Start of Outer Loop Over Weights
                        "1:\n\t"
                        "ld1 {v0.16b},  [%[input]], #16\n\t"
                        "ld1 {v1.16b},  [%[input]], #16\n\t"

                        // Saturate to 7
                        // CMGE AT, A, 7
                        "cmge v2.16b, v0.16b, v30.16b\n\t"
                        "cmge v3.16b, v1.16b, v30.16b\n\t"
                        // NOT AT, AT
                        "not v2.16b, v2.16b\n\t"
                        "not v3.16b, v3.16b\n\t"
                        // AND A, A, AT
                        "and v0.16b, v0.16b, v2.16b\n\t"
                        "and v1.16b, v1.16b, v3.16b\n\t"
                        // NOT ATT, AT
                        "not v2.16b, v2.16b\n\t"
                        "not v3.16b, v3.16b\n\t"
                        // AND ATTT, A, ATT
                        "and v2.16b, v30.16b, v2.16b\n\t"
                        "and v3.16b, v30.16b, v3.16b\n\t"
                        // ORR A, A, ATT
                        "orr v0.16b, v0.16b, v2.16b\n\t"
                        "orr v1.16b, v1.16b, v3.16b\n\t"
                        
                        // Saturate to -8
                        // CMGE AT, A, -8
                        "cmge v2.16b, v0.16b, v29.16b\n\t"
                        "cmge v3.16b, v1.16b, v29.16b\n\t"
                        // AND A, A, AT
                        "and v0.16b, v0.16b, v2.16b\n\t"
                        "and v1.16b, v1.16b, v3.16b\n\t"
                        // NOT ATT, AT
                        "not v2.16b, v2.16b\n\t"
                        "not v3.16b, v3.16b\n\t"
                        // AND ATTT, A, ATT
                        "and v2.16b, v29.16b, v2.16b\n\t"
                        "and v3.16b, v29.16b, v3.16b\n\t"
                        // ORR A, A, ATTT
                        "orr v0.16b, v0.16b, v2.16b\n\t"
                        "orr v1.16b, v1.16b, v3.16b\n\t"

                        // Pack 2 Saturated Int4 in 1 Int8
                        // AND AL, AL, 0x0F
                        "and v0.16b, v0.16b, v31.16b\n\t"
                        // SHL AH, AH, #4
                        "shl v1.16b, v1.16b, #4\n\t"
                        // ORR AM, AL, Ah
                        "orr v0.16b, v0.16b, v1.16b\n\t"
                        // ST1 AM, output
                        "st1 {v0.16b},  [%[output]], #16\n\t"

                        "add %w[i], %w[i], #32\n\t"
                        "cmp %w[i], %w[size]\n\t"
                        "b.lt 1b\n\t"

                        "sub %[input], %[input], %[size]\n\t"
                        "sub %[output], %[output], %[size], asr #1\n\t"

                        "3:\n\t"

                        : [ i ] "+r"(i)
                        : [ input ]  "r" (input), [ size ] "r"(size), [ output ] "r"(temp)
                        : "v0",  "v1",  "v2",  "v3",
                          "v28", "v29", "v30", "v31",
                          "w3",  "w4",  "w5",  "w6"
                    );
                    if (is_multibatched){
                        #ifdef W4A4_DECREASE_CONCURRENT_BATCH
                        doLowPrecision2BatchInputPack(temp, output, shape.size[0], shape.size[1] / 2);
                        #else
                        doLowPrecisionWeightPack(temp, output, shape.size[0], shape.size[1] / 2);
                        #endif
                        LowPrecision::deallocate(temp);
                    }
                }
                return Status::Success;
            }
            Status MultiplyInt8SingleBatch(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape
            ){
                int lhs_columns = input_shape.size[input_shape.number_dims - 1] ,
                    rhs_rows = kernel_shape.size[0] ,
                    rhs_columns = kernel_shape.size[1];
                if (lhs_columns != rhs_columns)
                    return Status::SizesMisMatch;
                if(lhs_columns == 0 || rhs_rows == 0)
                    return Status::Success;
                
                int8_t*         _kernel = const_cast<int8_t*>(kernel);
                const int8_t*   _input  = input;
                int32_t*        _output = output;
                int i, j;

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
                    
                    "cmp %w[size], #0\n\t"
                    "beq 3f\n\t"

                    // Start of Outer Loop Over Weights
                    "1:\n\t"
                    "ld1 {v0.16b},  [%[weights]], #16\n\t"
                    "ld1 {v1.16b},  [%[weights]], #16\n\t"
                    "ld1 {v2.16b},  [%[weights]], #16\n\t"
                    "ld1 {v3.16b},  [%[weights]], #16\n\t"

                    "ld1 {v4.16b},  [%[activation]], #16\n\t"

                    // SHL AL, A, #4
                    "shl v5.16b,  v4.16b,  #4\n\t"
                    // SSHR AH, A, #4
                    "sshr v4.16b, v4.16b, #4\n\t"
                    // SSHR AL, AL, #4
                    "sshr v5.16b,  v5.16b,  #4\n\t"

                    // SHL WL, W, #4
                    "shl v8.16b,  v0.16b, #4\n\t"
                    "shl v10.16b, v1.16b, #4\n\t"
                    "shl v11.16b, v2.16b, #4\n\t"
                    "shl v12.16b, v3.16b, #4\n\t"
                    
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

                    // SMULL T1.8h, WH.8b, A2.8b
                    "smull v13.8h, v0.8b, v5.8b\n\t"
                    "smull v14.8h, v1.8b, v5.8b\n\t"
                    "smull v15.8h, v2.8b, v5.8b\n\t"
                    "smull v16.8h, v3.8b, v5.8b\n\t"

                    // SMLAL2 T1.8h, WH.16b, A2.16b
                    "smlal2 v13.8h, v0.16b, v5.16b\n\t"
                    "smlal2 v14.8h, v1.16b, v5.16b\n\t"
                    "smlal2 v15.8h, v2.16b, v5.16b\n\t"
                    "smlal2 v16.8h, v3.16b, v5.16b\n\t"

                    // SMLAL T1.8h, WL.8b, A1.8b
                    "smlal v13.8h, v8.8b,  v4.8b\n\t"
                    "smlal v14.8h, v10.8b, v4.8b\n\t"
                    "smlal v15.8h, v11.8b, v4.8b\n\t"
                    "smlal v16.8h, v12.8b, v4.8b\n\t"

                    // SMLAL2 T1.8h, WL.16b, A1.16b
                    "smlal2 v13.8h, v8.16b,  v4.16b\n\t"
                    "smlal2 v14.8h, v10.16b, v4.16b\n\t"
                    "smlal2 v15.8h, v11.16b, v4.16b\n\t"
                    "smlal2 v16.8h, v12.16b, v4.16b\n\t"

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
                    "st1 {v23.4s},  [%[dst]], #16\n\t"

                    "mov %[activation], %[lhs_base]\n\t"

                    "3:\n\t"

                    "add %w[j], %w[j], #4\n\t"
                    "cmp %w[j], %w[rows]\n\t"
                    "b.lt 0b\n\t"


                    : [ dst ]        "+r" (_output),     [ i ]           "+r" (i),
                      [ j ]          "+r" (j)

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
                // std::cout << "Inside MultiplyInt8 with multibatch set to " 
                //           << ((input_shape.number_dims == 2 && input_shape.size[0] != 1)?("ON"):("OFF"))
                //           << std::endl;
                if (input_shape.number_dims == 2 && input_shape.size[0] != 1)
                    return MultiplyInt8MultiBatched(input, input_shape, kernel, kernel_shape, output, output_shape, LowPrecision::MulParams());
                return Status::NotUpdated;
                
                int lhs_columns = input_shape.size[input_shape.number_dims - 1] ,
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
                int32_t* output, Shape output_shape,
                LowPrecision::MulParams params
            ){
                int lhs_batches = input_shape.size[0],
                    lhs_columns = input_shape.size[1],
                    rhs_rows    = kernel_shape.size[0],
                    rhs_columns = kernel_shape.size[1];
                
                int need_downcasting = (params.need_downcasting)?(0xff):(0x00);
                
                if (lhs_columns != rhs_columns)
                    return Status::SizesMisMatch;
                if(lhs_columns == 0 || rhs_rows == 0 || lhs_batches == 0)
                    return Status::Success;
                if (lhs_batches % 4)
                    return Status::NotSupported;
                
                int8_t*         _kernel     = const_cast<int8_t*>(kernel);
                int8_t*         _kernel_base= const_cast<int8_t*>(kernel);
                int8_t*         _input      = const_cast<int8_t*>(input);
                int8_t*         _input_base = const_cast<int8_t*>(input);
                int i, j, k, end;
#ifdef W4A4_USE_16_BIT_ACCUMULATOR

                int32_t*        _output_1   = output + 0 * rhs_rows;
                int32_t*        _output_2   = output + 1 * rhs_rows;
                int32_t*        _output_3   = output + 2 * rhs_rows;
                int32_t*        _output_4   = output + 3 * rhs_rows;
                /* Vector assignments:
                    * W, WH     -> v0-3      (Weights, Weights High)
                    * A, AH     -> v4-7      (Activations, Activations High)
                    * WL        -> v8-11     (Weights Low)
                    * AL        -> v12-15    (Activations Low)
                    * ACC3      -> v16-19    (Accumulators input row #3)
                    * ACC4      -> v20-23    (Accumulators input row #4)
                    * ACC1      -> v24-27    (Accumulators input row #1)
                    * ACC2      -> v28-31    (Accumulators input row #2)
                */
                asm volatile(
                    "mov x1, %[activation]\n\t"
                    "mov x2, %[weights]\n\t"
                    "mov %w[k], wzr\n\t"

                    // Start of The Loop Over Batches
                    "5:\n\t"
                    "mov %w[j], wzr\n\t"

                    "0:\n\t"
                    "mov %w[i], wzr\n\t"
                    "movi v16.4s, #0\n\t"
                    "movi v17.4s, #0\n\t"
                    "movi v18.4s, #0\n\t"
                    "movi v19.4s, #0\n\t"
                    "movi v20.4s, #0\n\t"
                    "movi v21.4s, #0\n\t"
                    "movi v22.4s, #0\n\t"
                    "movi v23.4s, #0\n\t"
                    "movi v24.4s, #0\n\t"
                    "movi v25.4s, #0\n\t"
                    "movi v26.4s, #0\n\t"
                    "movi v27.4s, #0\n\t"
                    "movi v28.4s, #0\n\t"
                    "movi v29.4s, #0\n\t"
                    "movi v30.4s, #0\n\t"
                    "movi v31.4s, #0\n\t"

                    // Start of Outer Loop Over Weights

                    // Load Weights
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "ld1 {v0.16b, v1.16b, v2.16b, v3.16b},  [%[weights]]\n\t"
#else
                    "ld1 {v0.16b, v1.16b, v2.16b, v3.16b},  [%[weights]], #64\n\t"
#endif

                    // Load Activations
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "ld1 {v4.16b, v5.16b, v6.16b, v7.16b},  [%[activation]]\n\t"
#else
                    "ld1 {v4.16b, v5.16b, v6.16b, v7.16b},  [%[activation]], #64\n\t"
#endif

                    "1:\n\t"

                    // SHL WL, W, #4
                    "shl v8.16b, v0.16b,  #4\n\t"
                    "shl v9.16b, v1.16b,  #4\n\t"
                    "shl v10.16b, v2.16b, #4\n\t"
                    "shl v11.16b, v3.16b, #4\n\t"
                    
                    // SHL AL, A, #4
                    "shl v12.16b, v4.16b,  #4\n\t"
                    "shl v13.16b, v5.16b,  #4\n\t"
                    "shl v14.16b, v6.16b,  #4\n\t"
                    "shl v15.16b, v7.16b,  #4\n\t"

                    // SSHR WH, W, #4
                    "sshr v0.16b, v0.16b, #4\n\t"
                    "sshr v1.16b, v1.16b, #4\n\t"
                    "sshr v2.16b, v2.16b, #4\n\t"
                    "sshr v3.16b, v3.16b, #4\n\t"

                    // SSHR AH, A, #4
                    "sshr v4.16b, v4.16b, #4\n\t"
                    "sshr v5.16b, v5.16b, #4\n\t"
                    "sshr v6.16b, v6.16b, #4\n\t"
                    "sshr v7.16b, v7.16b, #4\n\t"
                    
                    // SSHR WL, WL, #4
                    "sshr v8.16b, v8.16b, #4\n\t"
                    "sshr v9.16b, v9.16b, #4\n\t"
                    "sshr v10.16b, v10.16b, #4\n\t"
                    "sshr v11.16b, v11.16b, #4\n\t"

                    // SSHR AL, AL, #4
                    "sshr v12.16b, v12.16b,  #4\n\t"
                    "sshr v13.16b, v13.16b,  #4\n\t"
                    "sshr v14.16b, v14.16b,  #4\n\t"
                    "sshr v15.16b, v15.16b,  #4\n\t"

                    // Activation Row #1
                    // SMLAL ACC1, WH, AH[0]
                    "smlal v24.8h, v0.8b, v4.8b\n\t"
                    "smlal v25.8h, v1.8b, v4.8b\n\t"
                    "smlal v26.8h, v2.8b, v4.8b\n\t"
                    "smlal v27.8h, v3.8b, v4.8b\n\t"

                    // Activation Row #2
                    // SMLAL ACC2, WH, AH[1]
                    "smlal v28.8h, v0.8b, v5.8b\n\t"
                    "smlal v29.8h, v1.8b, v5.8b\n\t"
                    "smlal v30.8h, v2.8b, v5.8b\n\t"
                    "smlal v31.8h, v3.8b, v5.8b\n\t"

                    // Activation Row #3
                    // SMLAL ACC3, WH, AH[2]
                    "smlal v16.8h, v0.8b, v6.8b\n\t"
                    "smlal v17.8h, v1.8b, v6.8b\n\t"
                    "smlal v18.8h, v2.8b, v6.8b\n\t"
                    "smlal v19.8h, v3.8b, v6.8b\n\t"

                    // Activation Row #4
                    // SMLAL ACC4, WH, AH[3]
                    "smlal v20.8h, v0.8b, v7.8b\n\t"
                    "smlal v21.8h, v1.8b, v7.8b\n\t"
                    "smlal v22.8h, v2.8b, v7.8b\n\t"
                    "smlal v23.8h, v3.8b, v7.8b\n\t"

                    // Activation Row #1
                    // SMLAL2 ACC1, WH, AH[0]
                    "smlal2 v24.8h, v0.16b, v4.16b\n\t"
                    "smlal2 v25.8h, v1.16b, v4.16b\n\t"
                    "smlal2 v26.8h, v2.16b, v4.16b\n\t"
                    "smlal2 v27.8h, v3.16b, v4.16b\n\t"

                    // Activation Row #2
                    // SMLAL2 ACC2, WH, AH[1]
                    "smlal2 v28.8h, v0.16b, v5.16b\n\t"
                    "smlal2 v29.8h, v1.16b, v5.16b\n\t"
                    "smlal2 v30.8h, v2.16b, v5.16b\n\t"
                    "smlal2 v31.8h, v3.16b, v5.16b\n\t"

                    // Activation Row #3
                    // SMLAL2 ACC3, WH, AH[2]
                    "smlal2 v16.8h, v0.16b, v6.16b\n\t"
                    "smlal2 v17.8h, v1.16b, v6.16b\n\t"
                    "smlal2 v18.8h, v2.16b, v6.16b\n\t"
                    "smlal2 v19.8h, v3.16b, v6.16b\n\t"

                    // Activation Row #4
                    // SMLAL2 ACC4, WH, AH[3]
                    "smlal2 v20.8h, v0.16b, v7.16b\n\t"
                    "smlal2 v21.8h, v1.16b, v7.16b\n\t"
                    "smlal2 v22.8h, v2.16b, v7.16b\n\t"
                    "smlal2 v23.8h, v3.16b, v7.16b\n\t"

                    // 
                    // Higher half of weight and activations vectors
                    // 

                    // Load Weights
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "ld1 {v0.16b, v1.16b, v2.16b, v3.16b},  [%[weights]]\n\t"
#else
                    "ld1 {v0.16b, v1.16b, v2.16b, v3.16b},  [%[weights]], #64\n\t"
#endif

                    // Load Activations
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "ld1 {v4.16b, v5.16b, v6.16b, v7.16b},  [%[activation]]\n\t"
#else
                    "ld1 {v4.16b, v5.16b, v6.16b, v7.16b},  [%[activation]], #64\n\t"
#endif

                    // Activation Row #1
                    // SMLAL ACC1, WL, AL[0]
                    "smlal v24.8h, v8.8b,  v12.8b\n\t"
                    "smlal v25.8h, v9.8b,  v12.8b\n\t"
                    "smlal v26.8h, v10.8b, v12.8b\n\t"
                    "smlal v27.8h, v11.8b, v12.8b\n\t"

                    // Activation Row #2
                    // SMLAL ACC2, WL, AL[1]
                    "smlal v28.8h, v8.8b,  v13.8b\n\t"
                    "smlal v29.8h, v9.8b,  v13.8b\n\t"
                    "smlal v30.8h, v10.8b, v13.8b\n\t"
                    "smlal v31.8h, v11.8b, v13.8b\n\t"

                    // Activation Row #3
                    // SMLAL ACC3, WL, AL[2]
                    "smlal v16.8h, v8.8b,  v14.8b\n\t"
                    "smlal v17.8h, v9.8b,  v14.8b\n\t"
                    "smlal v18.8h, v10.8b, v14.8b\n\t"
                    "smlal v19.8h, v11.8b, v14.8b\n\t"

                    // Activation Row #4
                    // SMLAL ACC4, WL, AL[3]
                    "smlal v20.8h, v8.8b,  v15.8b\n\t"
                    "smlal v21.8h, v9.8b,  v15.8b\n\t"
                    "smlal v22.8h, v10.8b, v15.8b\n\t"
                    "smlal v23.8h, v11.8b, v15.8b\n\t"

                    // Activation Row #1
                    // SMLAL2 ACC1, WL, AL[0]
                    "smlal2 v24.8h, v8.16b,  v12.16b\n\t"
                    "smlal2 v25.8h, v9.16b,  v12.16b\n\t"
                    "smlal2 v26.8h, v10.16b, v12.16b\n\t"
                    "smlal2 v27.8h, v11.16b, v12.16b\n\t"

                    // Activation Row #2
                    // SMLAL2 ACC2, WL, AL[1]
                    "smlal2 v28.8h, v8.16b,  v13.16b\n\t"
                    "smlal2 v29.8h, v9.16b,  v13.16b\n\t"
                    "smlal2 v30.8h, v10.16b, v13.16b\n\t"
                    "smlal2 v31.8h, v11.16b, v13.16b\n\t"

                    // Activation Row #3
                    // SMLAL2 ACC3, WL, AL[2]
                    "smlal2 v16.8h, v8.16b,  v14.16b\n\t"
                    "smlal2 v17.8h, v9.16b,  v14.16b\n\t"
                    "smlal2 v18.8h, v10.16b, v14.16b\n\t"
                    "smlal2 v19.8h, v11.16b, v14.16b\n\t"

                    // Activation Row #4
                    // SMLAL2 ACC4, WL, AL[3]
                    "smlal2 v20.8h, v8.16b,  v15.16b\n\t"
                    "smlal2 v21.8h, v9.16b,  v15.16b\n\t"
                    "smlal2 v22.8h, v10.16b, v15.16b\n\t"
                    "smlal2 v23.8h, v11.16b, v15.16b\n\t"

                    // Check if the loop over rows of weight matrix is done
                    "add %w[i], %w[i], #32\n\t"
                    "cmp %w[i], %w[size]\n\t"
                    "b.lt 1b\n\t"

                    // SADDLP ACC1, ACC1
                    "sadalp v24.4s, v24.8h\n\t"
                    "sadalp v25.4s, v25.8h\n\t"
                    "sadalp v26.4s, v26.8h\n\t"
                    "sadalp v27.4s, v27.8h\n\t"
                    
                    // SADDLP ACC2, ACC2
                    "sadalp v28.4s, v28.8h\n\t"
                    "sadalp v29.4s, v29.8h\n\t"
                    "sadalp v30.4s, v30.8h\n\t"
                    "sadalp v31.4s, v31.8h\n\t"

                    // SADDLP ACC3, ACC3
                    "sadalp v16.4s, v16.8h\n\t"
                    "sadalp v17.4s, v17.8h\n\t"
                    "sadalp v18.4s, v18.8h\n\t"
                    "sadalp v19.4s, v19.8h\n\t"
                    
                    // SADDLP ACC4, ACC4
                    "sadalp v20.4s, v20.8h\n\t"
                    "sadalp v21.4s, v21.8h\n\t"
                    "sadalp v22.4s, v22.8h\n\t"
                    "sadalp v23.4s, v23.8h\n\t"

                    // Accumulate the ACC1 to one int32
                    "addv s24, v24.4s\n\t"
                    "addv s25, v25.4s\n\t"
                    "addv s26, v26.4s\n\t"
                    "addv s27, v27.4s\n\t"

                    // Accumulate the ACC2 to one int32
                    "addv s28, v28.4s\n\t"
                    "addv s29, v29.4s\n\t"
                    "addv s30, v30.4s\n\t"
                    "addv s31, v31.4s\n\t"

                    // Accumulate the ACC3 to one int32
                    "addv s16, v16.4s\n\t"
                    "addv s17, v17.4s\n\t"
                    "addv s18, v18.4s\n\t"
                    "addv s19, v19.4s\n\t"

                    // Accumulate the ACC4 to one int32
                    "addv s20, v20.4s\n\t"
                    "addv s21, v21.4s\n\t"
                    "addv s22, v22.4s\n\t"
                    "addv s23, v23.4s\n\t"

                    // Reorder ACC3 to store
                    "mov v16.s[1], v17.s[0]\n\t"
                    "mov v16.s[2], v18.s[0]\n\t"
                    "mov v16.s[3], v19.s[0]\n\t"

                    // Reorder ACC4 to store
                    "mov v20.s[1], v21.s[0]\n\t"
                    "mov v20.s[2], v22.s[0]\n\t"
                    "mov v20.s[3], v23.s[0]\n\t"

                    // Reorder ACC1 to store
                    "mov v24.s[1], v25.s[0]\n\t"
                    "mov v24.s[2], v26.s[0]\n\t"
                    "mov v24.s[3], v27.s[0]\n\t"

                    // Reorder ACC2 to store
                    "mov v28.s[1], v29.s[0]\n\t"
                    "mov v28.s[2], v30.s[0]\n\t"
                    "mov v28.s[3], v31.s[0]\n\t"

                    // Check if the output need downcasting to int8
                    "cmp %w[downcast], 0xff\n\t"
                    "beq 6f\n\t"

                    // Output is needed in 32-bit; no need to downcast
                    // Store the 4 int32 results
                    "st1 {v24.4s},  [%[dst_3]], #16\n\t"
                    "st1 {v28.4s},  [%[dst_4]], #16\n\t"
                    "st1 {v16.4s},  [%[dst_1]], #16\n\t"
                    "st1 {v20.4s},  [%[dst_2]], #16\n\t"

                    // Jump to after dowcasting since we dont need to downcast
                    "b 7f\n\t"

                    // Need to Downcast to Int8
                    "6:\n\t"

                    // Cast 32-bit to 16-bit
                    "sqxtn v24.4h, v24.4s\n\t"
                    "sqxtn v28.4h, v28.4s\n\t"
                    "sqxtn v16.4h, v16.4s\n\t"
                    "sqxtn v20.4h, v20.4s\n\t"

                    // Cast 16-bit to 8-bit
                    "sqxtn v24.8b, v24.8h\n\t"
                    "sqxtn v28.8b, v28.8h\n\t"
                    "sqxtn v16.8b, v16.8h\n\t"
                    "sqxtn v20.8b, v20.8h\n\t"
                    
                    // Store the 4 int8 results
                    "st1 {v24.s}[0],  [%[dst_3]], #4\n\t"
                    "st1 {v28.s}[0],  [%[dst_4]], #4\n\t"
                    "st1 {v16.s}[0],  [%[dst_1]], #4\n\t"
                    "st1 {v20.s}[0],  [%[dst_2]], #4\n\t"

                    "7:\n\t"

                    // Reset the activations to the start of the row
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "mov %[activation], %[activation]\n\t"
#else
                    "mov %[activation], x1\n\t"
#endif

                    // Check if the all the columns of weight matrix are processed
                    "add %w[j], %w[j], #4\n\t"
                    "cmp %w[j], %w[rows]\n\t"
                    "b.lt 0b\n\t"

                    // Prepare the activation base for next 4 batches
                    "add x1, x1, %[size], asr #2\n\t"
                    
                    // Reset the activations to the start of the row
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "mov %[activation], %[activation]\n\t"
#else
                    "mov %[activation], x1\n\t"
#endif

                    // Reset the weights to the start
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "mov %[weights], %[weights]\n\t"
#else
                    "mov %[weights], x2\n\t"
#endif

                    // Check if the output needed downcasting to int8
                    "cmp %w[downcast], 0xff\n\t"
                    "beq 8f\n\t"

                    // Prepare the destination base for next 4 batches
                    "mov %[dst_1], %[dst_3]\n\t"
                    "add %[dst_2], %[dst_1], %[rows], lsl #2\n\t"
                    "add %[dst_3], %[dst_2], %[rows], lsl #2\n\t"
                    "add %[dst_4], %[dst_3], %[rows], lsl #2\n\t"

                    // Jump to after the case for downcasting
                    "b 9f\n\t"

                    // Needed to Downcast to Int8
                    "8:\n\t"

                    // Prepare the destination base for next 4 batches
                    "mov %[dst_1], %[dst_3]\n\t"
                    "add %[dst_2], %[dst_1], %[rows]\n\t"
                    "add %[dst_3], %[dst_2], %[rows]\n\t"
                    "add %[dst_4], %[dst_3], %[rows]\n\t"

                    "9:\n\t"

                    // Check if the all the columns of weight matrix are processed
                    "add %w[k], %w[k], #4\n\t"
                    "cmp %w[k], %w[batches]\n\t"
                    "b.lt 5b\n\t"


                    : [ dst_1 ]      "+r" (_output_1),   [ dst_2 ]       "+r" (_output_2),
                      [ dst_3 ]      "+r" (_output_3),   [ dst_4 ]       "+r" (_output_4),
                      [ i ]          "+r" (i),           [ end ]         "+r" (end),
                      [ j ]          "+r" (j),           [ k ]           "+r" (k)

                    : [ activation ] "r"  (_input),      [ act_base ]    "r"  (_input_base),
                      [ weights ]    "r"  (_kernel),     [ wts_base ]    "r"  (_kernel_base),
                      [ size ]       "r"  (lhs_columns), [ rows ]        "r"  (rhs_rows),
                      [ batches ]    "r"  (lhs_batches), [ downcast ]    "r"  (need_downcasting)

                    : "v0",  "v1",  "v2",  "v3",
                      "v4",  "v5",  "v6",  "v7",
                      "v8",  "v9",  "v10", "v11",
                      "v12", "v13", "v14", "v15",
                      "v16", "v17", "v18", "v19",
                      "v20", "v21", "v22", "v23",
                      "v24", "v25", "v26", "v27",
                      "v28", "v29", "v30", "v31",
                      "x0" , "x1" , "x2"
                );
#else
#ifdef W4A4_USE_SINGLE_BATCH_FOR_MULTI
                int32_t*        _output_1   = output + 0 * rhs_rows;
                /* Vector assignments:
                    * W, WH     -> v0-3      (Weights, Weights High)
                    * A, AH     -> v4        (Activations, Activations High)
                    * AL        -> v5        (Activations Low)
                    *  -        -> v6-7      (Not Used)
                    * WL        -> v8-11     (Activations Low)
                    * MiniACC   -> v12-15    (Weights Low)
                    *  -        -> v16-19    (Not Used)
                    *  -        -> v20-22    (Not Used)
                    *  -        -> v26-27    (Not Used)
                    * ACC       -> v28-31    (Accumulators)
                */
                asm volatile(
                    "mov %w[k], wzr\n\t"

                    // Start of The Loop Over Batches
                    "5:\n\t"
                    "mov %w[j], wzr\n\t"

                    // "cmp %w[rows], #0\n\t"
                    // "beq 5f\n\t"

                    "0:\n\t"
                    "mov %w[i], wzr\n\t"
                    "dup v23.4s, wzr\n\t"
                    "dup v24.4s, wzr\n\t"
                    "dup v25.4s, wzr\n\t"
                    "dup v30.4s, wzr\n\t"
                    
                    "cmp %w[size], #0\n\t"
                    "beq 3f\n\t"

                    // Start of Outer Loop Over Weights
                    "1:\n\t"
                    "ld1 {v0.16b},  [%[weights]], #16\n\t"
                    "ld1 {v1.16b},  [%[weights]], #16\n\t"
                    "ld1 {v2.16b},  [%[weights]], #16\n\t"
                    "ld1 {v3.16b},  [%[weights]], #16\n\t"

                    "ld1 {v4.16b},  [%[activation]], #16\n\t"

                    // SHL AL, A, #4
                    "shl v5.16b,  v4.16b,  #4\n\t"
                    // SSHR AH, A, #4
                    "sshr v4.16b, v4.16b, #4\n\t"
                    // SSHR AL, AL, #4
                    "sshr v5.16b,  v5.16b,  #4\n\t"

                    // SHL WL, W, #4
                    "shl v8.16b,  v0.16b, #4\n\t"
                    "shl v10.16b, v1.16b, #4\n\t"
                    "shl v11.16b, v2.16b, #4\n\t"
                    "shl v12.16b, v3.16b, #4\n\t"
                    
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

                    // SMULL MiniACC.8h, WH.8b, A2.8b
                    "smull v12.8h, v0.8b, v5.8b\n\t"
                    "smull v13.8h, v1.8b, v5.8b\n\t"
                    "smull v14.8h, v2.8b, v5.8b\n\t"
                    "smull v15.8h, v3.8b, v5.8b\n\t"

                    // SMLAL2 MiniACC.8h, WH.16b, A2.16b
                    "smlal2 v12.8h, v0.16b, v5.16b\n\t"
                    "smlal2 v13.8h, v1.16b, v5.16b\n\t"
                    "smlal2 v14.8h, v2.16b, v5.16b\n\t"
                    "smlal2 v15.8h, v3.16b, v5.16b\n\t"

                    // SMLAL MiniACC.8h, WL.8b, A1.8b
                    "smlal v12.8h, v8.8b,  v4.8b\n\t"
                    "smlal v13.8h, v10.8b, v4.8b\n\t"
                    "smlal v14.8h, v11.8b, v4.8b\n\t"
                    "smlal v15.8h, v12.8b, v4.8b\n\t"

                    // SMLAL2 MiniACC.8h, WL.16b, A1.16b
                    "smlal2 v12.8h, v8.16b,  v4.16b\n\t"
                    "smlal2 v13.8h, v10.16b, v4.16b\n\t"
                    "smlal2 v14.8h, v11.16b, v4.16b\n\t"
                    "smlal2 v15.8h, v12.16b, v4.16b\n\t"

                    // ACCUMULATE ACC, MiniACC
                    "sadalp v28.4s, v12.8h\n\t"
                    "sadalp v29.4s, v13.8h\n\t"
                    "sadalp v30.4s, v14.8h\n\t"
                    "sadalp v31.4s, v15.8h\n\t"

                    "add %w[i], %w[i], #32\n\t"
                    "cmp %w[i], %w[size]\n\t"
                    "b.lt 1b\n\t"

                    "addv s23, v28.4s\n\t"
                    "addv s24, v29.4s\n\t"
                    "addv s25, v30.4s\n\t"
                    "addv s30, v31.4s\n\t"

                    "mov v28.s[1], v29.s[0]\n\t"
                    "mov v28.s[2], v30.s[0]\n\t"
                    "mov v28.s[3], v31.s[0]\n\t"

                    // "ld1 {v30.4s},  [%[dst]]\n\t"
                    // "add v30.4s, v30.4s, v28.4s\n\t"
                    // "st1 {v30.4s},  [%[dst]]\n\t"
                    "st1 {v28.4s},  [%[dst_1]], #16\n\t"

                    "mov %[activation], %[act_base]\n\t"

                    "3:\n\t"

                    "add %w[j], %w[j], #4\n\t"
                    "cmp %w[j], %w[rows]\n\t"
                    "b.lt 0b\n\t"

                    // Prepare the activation base for next 4 batches
                    "add %[act_base], %[act_base], %[size], asr #1\n\t"
                    
                    // Reset the activations to the start of the row
                    "mov %[activation], %[act_base]\n\t"

                    // Reset the weights to the start
                    "mov %[weights], %[wts_base]\n\t"

                    // Check if the all the columns of weight matrix are processed
                    "add %w[k], %w[k], #1\n\t"
                    "cmp %w[k], %w[batches]\n\t"
                    "b.lt 5b\n\t"


                    : [ dst_1 ]      "+r" (_output_1),
                      [ i ]          "+r" (i),           [ end ]         "+r" (end),
                      [ j ]          "+r" (j),           [ k ]           "+r" (k)

                    : [ activation ] "r"  (_input),      [ act_base ]    "r"  (_input_base),
                      [ weights ]    "r"  (_kernel),     [ wts_base ]    "r"  (_kernel_base),
                      [ size ]       "r"  (lhs_columns), [ rows ]        "r"  (rhs_rows),
                      [ batches ]    "r"  (lhs_batches)

                    : "v0",  "v1",  "v2",  "v3",
                      "v4",  "v5",  "v6",  "v7",
                      "v8",  "v9",  "v10", "v11",
                      "v12", "v13", "v14", "v15",
                      "v16", "v17", "v18", "v19",
                      "v20", "v21", "v22", "v23",
                      "v24", "v25", "v26", "v27",
                      "v28", "v29", "v30", "v31"
                );
#else
#ifndef W4A4_DECREASE_CONCURRENT_BATCH
                int32_t*        _output_1   = output + 0 * rhs_rows;
                int32_t*        _output_2   = output + 1 * rhs_rows;
                int32_t*        _output_3   = output + 2 * rhs_rows;
                int32_t*        _output_4   = output + 3 * rhs_rows;
                /* Vector assignments:
                    * W, WH     -> v0-3      (Weights, Weights High)
                    * A, AH     -> v4-5      (Activations, Activations High)
                    * AL        -> v6-7      (Activations Low)
                    * WL        -> v8-11     (Activations Low)
                    * MiniACC   -> v12-15    (Weights Low)
                    * ACC3      -> v16-19    (Accumulators input row #3)
                    * ACC4      -> v20-23    (Accumulators input row #4)
                    * ACC1      -> v24-27    (Accumulators input row #1)
                    * ACC2      -> v28-31    (Accumulators input row #2)
                */
                asm volatile(
                    "mov x1, %[activation]\n\t"
                    "mov x2, %[weights]\n\t"
                    "mov %w[k], wzr\n\t"

                    // Start of The Loop Over Batches
                    "5:\n\t"
                    "mov %w[j], wzr\n\t"

                    "0:\n\t"
                    "mov %w[i], wzr\n\t"
                    "movi v16.4s, #0\n\t"
                    "movi v17.4s, #0\n\t"
                    "movi v18.4s, #0\n\t"
                    "movi v19.4s, #0\n\t"
                    "movi v20.4s, #0\n\t"
                    "movi v21.4s, #0\n\t"
                    "movi v22.4s, #0\n\t"
                    "movi v23.4s, #0\n\t"
                    "movi v24.4s, #0\n\t"
                    "movi v25.4s, #0\n\t"
                    "movi v26.4s, #0\n\t"
                    "movi v27.4s, #0\n\t"
                    "movi v28.4s, #0\n\t"
                    "movi v29.4s, #0\n\t"
                    "movi v30.4s, #0\n\t"
                    "movi v31.4s, #0\n\t"

                    // Start of Outer Loop Over Weights

                    
                    "1:\n\t"
                    // Load Weights
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "ld1 {v0.16b, v1.16b, v2.16b, v3.16b},  [%[weights]]\n\t"
#else
                    "ld1 {v0.16b, v1.16b, v2.16b, v3.16b},  [%[weights]], #64\n\t"
#endif
                    // "ld1 {v0.16b},  [%[weights]], #16\n\t"
                    // "ld1 {v1.16b},  [%[weights]], #16\n\t"
                    // "ld1 {v2.16b},  [%[weights]], #16\n\t"
                    // "ld1 {v3.16b},  [%[weights]], #16\n\t"

                    // Load Activations
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "ld1 {v4.16b, v5.16b},  [%[activation]]\n\t"
#else
                    "ld1 {v4.16b, v5.16b},  [%[activation]], #32\n\t"
#endif
                    // "ld1 {v4.16b},  [%[activation]], #16\n\t"
                    // "ld1 {v5.16b},  [%[activation]], #16\n\t"

                    // SHL WL, W, #4
                    "shl v8.16b, v0.16b,  #4\n\t"
                    "shl v9.16b, v1.16b,  #4\n\t"
                    "shl v10.16b, v2.16b, #4\n\t"
                    "shl v11.16b, v3.16b, #4\n\t"
                    
                    // SSHR WL, WL, #4
                    "sshr v8.16b, v8.16b, #4\n\t"
                    "sshr v9.16b, v9.16b, #4\n\t"
                    "sshr v10.16b, v10.16b, #4\n\t"
                    "sshr v11.16b, v11.16b, #4\n\t"

                    // SSHR WH, W, #4
                    "sshr v0.16b, v0.16b, #4\n\t"
                    "sshr v1.16b, v1.16b, #4\n\t"
                    "sshr v2.16b, v2.16b, #4\n\t"
                    "sshr v3.16b, v3.16b, #4\n\t"
                    
                    // SHL AL, A, #4
                    "shl v6.16b,  v4.16b,  #4\n\t"
                    "shl v7.16b,  v5.16b,  #4\n\t"

                    // SSHR AH, A, #4
                    "sshr v4.16b, v4.16b, #4\n\t"
                    "sshr v5.16b, v5.16b, #4\n\t"

                    // SSHR AL, AL, #4
                    "sshr v6.16b,  v6.16b,  #4\n\t"
                    "sshr v7.16b,  v7.16b,  #4\n\t"

                    // // Activation Row #1
                    // // SMULL MiniACC, W, AL[0]
                    // "smull v12.8h, v0.8b, v8.8b\n\t"
                    // "smull v13.8h, v1.8b, v8.8b\n\t"
                    // "smull v14.8h, v2.8b, v8.8b\n\t"
                    // "smull v15.8h, v3.8b, v8.8b\n\t"
                    //
                    // // SMLAL2 MiniACC, W, AL[0]
                    // "smlal2 v12.8h, v0.16b, v8.16b\n\t"
                    // "smlal2 v13.8h, v1.16b, v8.16b\n\t"
                    // "smlal2 v14.8h, v2.16b, v8.16b\n\t"
                    // "smlal2 v15.8h, v3.16b, v8.16b\n\t"
                    //
                    // // SADALP ACC1, MiniACC
                    // "sadalp v16.4s, v12.8h\n\t"
                    // "sadalp v17.4s, v13.8h\n\t"
                    // "sadalp v18.4s, v14.8h\n\t"
                    // "sadalp v19.4s, v15.8h\n\t"
                    //
                    // // Activation Row #2
                    // // SMULL MiniACC, W, AL[1]
                    // "smull v12.8h, v0.8b, v9.8b\n\t"
                    // "smull v13.8h, v1.8b, v9.8b\n\t"
                    // "smull v14.8h, v2.8b, v9.8b\n\t"
                    // "smull v15.8h, v3.8b, v9.8b\n\t"
                    //
                    // // SMLAL2 MiniACC, W, AL[1]
                    // "smlal2 v12.8h, v0.16b, v9.16b\n\t"
                    // "smlal2 v13.8h, v1.16b, v9.16b\n\t"
                    // "smlal2 v14.8h, v2.16b, v9.16b\n\t"
                    // "smlal2 v15.8h, v3.16b, v9.16b\n\t"
                    //
                    // // SADALP ACC2, MiniACC
                    // "sadalp v20.4s, v12.8h\n\t"
                    // "sadalp v21.4s, v13.8h\n\t"
                    // "sadalp v22.4s, v14.8h\n\t"
                    // "sadalp v23.4s, v15.8h\n\t"

                    // Activation Row #1
                    // SMULL MiniACC, WL, AL[0]
                    "smull v12.8h, v0.8b, v6.8b\n\t"
                    "smull v13.8h, v1.8b, v6.8b\n\t"
                    "smull v14.8h, v2.8b, v6.8b\n\t"
                    "smull v15.8h, v3.8b, v6.8b\n\t"

                    // SMLAL2 MiniACC, WL, AL[0]
                    "smlal2 v12.8h, v0.16b, v6.16b\n\t"
                    "smlal2 v13.8h, v1.16b, v6.16b\n\t"
                    "smlal2 v14.8h, v2.16b, v6.16b\n\t"
                    "smlal2 v15.8h, v3.16b, v6.16b\n\t"

                    // SADALP ACC1, MiniACC
                    "sadalp v24.4s, v12.8h\n\t"
                    "sadalp v25.4s, v13.8h\n\t"
                    "sadalp v26.4s, v14.8h\n\t"
                    "sadalp v27.4s, v15.8h\n\t"

                    // Activation Row #2
                    // SMULL MiniACC, WL, AL[1]
                    "smull v12.8h, v0.8b, v7.8b\n\t"
                    "smull v13.8h, v1.8b, v7.8b\n\t"
                    "smull v14.8h, v2.8b, v7.8b\n\t"
                    "smull v15.8h, v3.8b, v7.8b\n\t"

                    // SMLAL2 MiniACC, WL, AL[1]
                    "smlal2 v12.8h, v0.16b, v7.16b\n\t"
                    "smlal2 v13.8h, v1.16b, v7.16b\n\t"
                    "smlal2 v14.8h, v2.16b, v7.16b\n\t"
                    "smlal2 v15.8h, v3.16b, v7.16b\n\t"

                    // SADALP ACC2, MiniACC
                    "sadalp v28.4s, v12.8h\n\t"
                    "sadalp v29.4s, v13.8h\n\t"
                    "sadalp v30.4s, v14.8h\n\t"
                    "sadalp v31.4s, v15.8h\n\t"

                    // 
                    // Higher half of weight vectors
                    // 

                    // // Activation Row #1
                    // // SMULL MiniACC, W, AH[0]
                    // "smull v12.8h, v0.8b, v4.8b\n\t"
                    // "smull v13.8h, v1.8b, v4.8b\n\t"
                    // "smull v14.8h, v2.8b, v4.8b\n\t"
                    // "smull v15.8h, v3.8b, v4.8b\n\t"
                    //
                    // // SMLAL2 MiniACC, W, AH[0]
                    // "smlal2 v12.8h, v0.16b, v4.16b\n\t"
                    // "smlal2 v13.8h, v1.16b, v4.16b\n\t"
                    // "smlal2 v14.8h, v2.16b, v4.16b\n\t"
                    // "smlal2 v15.8h, v3.16b, v4.16b\n\t"
                    //
                    // // SADALP ACC1, MiniACC
                    // "sadalp v16.4s, v12.8h\n\t"
                    // "sadalp v17.4s, v13.8h\n\t"
                    // "sadalp v18.4s, v14.8h\n\t"
                    // "sadalp v19.4s, v15.8h\n\t"
                    //
                    // // Activation Row #2
                    // // SMULL MiniACC, W, AH[1]
                    // "smull v12.8h, v0.8b, v5.8b\n\t"
                    // "smull v13.8h, v1.8b, v5.8b\n\t"
                    // "smull v14.8h, v2.8b, v5.8b\n\t"
                    // "smull v15.8h, v3.8b, v5.8b\n\t"
                    //
                    // // SMLAL2 MiniACC, W, AH[1]
                    // "smlal2 v12.8h, v0.16b, v5.16b\n\t"
                    // "smlal2 v13.8h, v1.16b, v5.16b\n\t"
                    // "smlal2 v14.8h, v2.16b, v5.16b\n\t"
                    // "smlal2 v15.8h, v3.16b, v5.16b\n\t"
                    //
                    // // SADALP ACC2, MiniACC
                    // "sadalp v20.4s, v12.8h\n\t"
                    // "sadalp v21.4s, v13.8h\n\t"
                    // "sadalp v22.4s, v14.8h\n\t"
                    // "sadalp v23.4s, v15.8h\n\t"

                    // Activation Row #1
                    // SMULL MiniACC, WH, AH[0]
                    "smull v12.8h, v8.8b,  v4.8b\n\t"
                    "smull v13.8h, v9.8b,  v4.8b\n\t"
                    "smull v14.8h, v10.8b, v4.8b\n\t"
                    "smull v15.8h, v11.8b, v4.8b\n\t"

                    // SMLAL2 MiniACC, WH, AH[0]
                    "smlal2 v12.8h, v8.16b,  v4.16b\n\t"
                    "smlal2 v13.8h, v9.16b,  v4.16b\n\t"
                    "smlal2 v14.8h, v10.16b, v4.16b\n\t"
                    "smlal2 v15.8h, v11.16b, v4.16b\n\t"

                    // SADALP ACC1, MiniACC
                    "sadalp v24.4s, v12.8h\n\t"
                    "sadalp v25.4s, v13.8h\n\t"
                    "sadalp v26.4s, v14.8h\n\t"
                    "sadalp v27.4s, v15.8h\n\t"

                    // Activation Row #2
                    // SMULL MiniACC, WH, AH[1]
                    "smull v12.8h, v8.8b,  v5.8b\n\t"
                    "smull v13.8h, v9.8b,  v5.8b\n\t"
                    "smull v14.8h, v10.8b, v5.8b\n\t"
                    "smull v15.8h, v11.8b, v5.8b\n\t"

                    // SMLAL2 MiniACC, WH, AH[1]
                    "smlal2 v12.8h, v8.16b,  v5.16b\n\t"
                    "smlal2 v13.8h, v9.16b,  v5.16b\n\t"
                    "smlal2 v14.8h, v10.16b, v5.16b\n\t"
                    "smlal2 v15.8h, v11.16b, v5.16b\n\t"

                    // SADALP ACC2, MiniACC
                    "sadalp v28.4s, v12.8h\n\t"
                    "sadalp v29.4s, v13.8h\n\t"
                    "sadalp v30.4s, v14.8h\n\t"
                    "sadalp v31.4s, v15.8h\n\t"
                    
                    // Load Activations
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "ld1 {v4.16b, v5.16b},  [%[activation]]\n\t"
#else
                    "ld1 {v4.16b, v5.16b},  [%[activation]], #32\n\t"
#endif
                    // "ld1 {v4.16b},  [%[activation]], #16\n\t"
                    // "ld1 {v5.16b},  [%[activation]], #16\n\t"

                    // SHL AL, A, #4
                    "shl v6.16b,  v4.16b,  #4\n\t"
                    "shl v7.16b,  v5.16b,  #4\n\t"

                    // SSHR AH, A, #4
                    "sshr v4.16b, v4.16b, #4\n\t"
                    "sshr v5.16b, v5.16b, #4\n\t"

                    // SSHR AL, AL, #4
                    "sshr v6.16b,  v6.16b,  #4\n\t"
                    "sshr v7.16b,  v7.16b,  #4\n\t"

                    // Activation Row #1
                    // SMULL MiniACC, WL, AL[0]
                    "smull v12.8h, v0.8b, v6.8b\n\t"
                    "smull v13.8h, v1.8b, v6.8b\n\t"
                    "smull v14.8h, v2.8b, v6.8b\n\t"
                    "smull v15.8h, v3.8b, v6.8b\n\t"

                    // SMLAL2 MiniACC, WL, AL[0]
                    "smlal2 v12.8h, v0.16b, v6.16b\n\t"
                    "smlal2 v13.8h, v1.16b, v6.16b\n\t"
                    "smlal2 v14.8h, v2.16b, v6.16b\n\t"
                    "smlal2 v15.8h, v3.16b, v6.16b\n\t"

                    // SADALP ACC3, MiniACC
                    "sadalp v16.4s, v12.8h\n\t"
                    "sadalp v17.4s, v13.8h\n\t"
                    "sadalp v18.4s, v14.8h\n\t"
                    "sadalp v19.4s, v15.8h\n\t"

                    // Activation Row #2
                    // SMULL MiniACC, WL, AL[1]
                    "smull v12.8h, v0.8b, v7.8b\n\t"
                    "smull v13.8h, v1.8b, v7.8b\n\t"
                    "smull v14.8h, v2.8b, v7.8b\n\t"
                    "smull v15.8h, v3.8b, v7.8b\n\t"

                    // SMLAL2 MiniACC, WL, AL[1]
                    "smlal2 v12.8h, v0.16b, v7.16b\n\t"
                    "smlal2 v13.8h, v1.16b, v7.16b\n\t"
                    "smlal2 v14.8h, v2.16b, v7.16b\n\t"
                    "smlal2 v15.8h, v3.16b, v7.16b\n\t"

                    // SADALP ACC4, MiniACC
                    "sadalp v20.4s, v12.8h\n\t"
                    "sadalp v21.4s, v13.8h\n\t"
                    "sadalp v22.4s, v14.8h\n\t"
                    "sadalp v23.4s, v15.8h\n\t"

                    // 
                    // Higher half of weight vectors
                    // 

                    // Activation Row #1
                    // SMULL MiniACC, WH, AH[0]
                    "smull v12.8h, v8.8b,  v4.8b\n\t"
                    "smull v13.8h, v9.8b,  v4.8b\n\t"
                    "smull v14.8h, v10.8b, v4.8b\n\t"
                    "smull v15.8h, v11.8b, v4.8b\n\t"

                    // SMLAL2 MiniACC, WH, AH[0]
                    "smlal2 v12.8h, v8.16b,  v4.16b\n\t"
                    "smlal2 v13.8h, v9.16b,  v4.16b\n\t"
                    "smlal2 v14.8h, v10.16b, v4.16b\n\t"
                    "smlal2 v15.8h, v11.16b, v4.16b\n\t"

                    // SADALP ACC3, MiniACC
                    "sadalp v16.4s, v12.8h\n\t"
                    "sadalp v17.4s, v13.8h\n\t"
                    "sadalp v18.4s, v14.8h\n\t"
                    "sadalp v19.4s, v15.8h\n\t"

                    // Activation Row #2
                    // SMULL MiniACC, WH, AH[1]
                    "smull v12.8h, v8.8b,  v5.8b\n\t"
                    "smull v13.8h, v9.8b,  v5.8b\n\t"
                    "smull v14.8h, v10.8b, v5.8b\n\t"
                    "smull v15.8h, v11.8b, v5.8b\n\t"

                    // SMLAL2 MiniACC, WH, AH[1]
                    "smlal2 v12.8h, v8.16b,  v5.16b\n\t"
                    "smlal2 v13.8h, v9.16b,  v5.16b\n\t"
                    "smlal2 v14.8h, v10.16b, v5.16b\n\t"
                    "smlal2 v15.8h, v11.16b, v5.16b\n\t"

                    // SADALP ACC4, MiniACC
                    "sadalp v20.4s, v12.8h\n\t"
                    "sadalp v21.4s, v13.8h\n\t"
                    "sadalp v22.4s, v14.8h\n\t"
                    "sadalp v23.4s, v15.8h\n\t"

                    // Check if the loop over rows of weight matrix is done
                    "add %w[i], %w[i], #32\n\t"
                    "cmp %w[i], %w[size]\n\t"
                    "b.lt 1b\n\t"

                    // Accumulate the ACC3 to one int32
                    "addv s16, v16.4s\n\t"
                    "addv s17, v17.4s\n\t"
                    "addv s18, v18.4s\n\t"
                    "addv s19, v19.4s\n\t"

                    // Accumulate the ACC4 to one int32
                    "addv s20, v20.4s\n\t"
                    "addv s21, v21.4s\n\t"
                    "addv s22, v22.4s\n\t"
                    "addv s23, v23.4s\n\t"

                    // Accumulate the ACC1 to one int32
                    "addv s24, v24.4s\n\t"
                    "addv s25, v25.4s\n\t"
                    "addv s26, v26.4s\n\t"
                    "addv s27, v27.4s\n\t"

                    // Accumulate the ACC2 to one int32
                    "addv s28, v28.4s\n\t"
                    "addv s29, v29.4s\n\t"
                    "addv s30, v30.4s\n\t"
                    "addv s31, v31.4s\n\t"

                    // Reorder ACC3 to store
                    "mov v16.s[1], v17.s[0]\n\t"
                    "mov v16.s[2], v18.s[0]\n\t"
                    "mov v16.s[3], v19.s[0]\n\t"

                    // Reorder ACC4 to store
                    "mov v20.s[1], v21.s[0]\n\t"
                    "mov v20.s[2], v22.s[0]\n\t"
                    "mov v20.s[3], v23.s[0]\n\t"

                    // Reorder ACC1 to store
                    "mov v24.s[1], v25.s[0]\n\t"
                    "mov v24.s[2], v26.s[0]\n\t"
                    "mov v24.s[3], v27.s[0]\n\t"

                    // Reorder ACC2 to store
                    "mov v28.s[1], v29.s[0]\n\t"
                    "mov v28.s[2], v30.s[0]\n\t"
                    "mov v28.s[3], v31.s[0]\n\t"
                    
                    // Store the 4 int32 results
                    "st1 {v24.4s},  [%[dst_3]], #16\n\t"
                    "st1 {v28.4s},  [%[dst_4]], #16\n\t"
                    "st1 {v16.4s},  [%[dst_1]], #16\n\t"
                    "st1 {v20.4s},  [%[dst_2]], #16\n\t"
                    
                    // Reset the activations to the start of the row
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "mov %[activation], %[activation]\n\t"
#else
                    "mov %[activation], x1\n\t"
#endif

                    // Check if the all the columns of weight matrix are processed
                    "add %w[j], %w[j], #4\n\t"
                    "cmp %w[j], %w[rows]\n\t"
                    "b.lt 0b\n\t"

                    // Prepare the activation base for next 4 batches
                    "add x1, x1, %[size], asr #2\n\t"
                    
                    // Reset the activations to the start of the row
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "mov %[activation], %[activation]\n\t"
#else
                    "mov %[activation], x1\n\t"
#endif

                    // Reset the weights to the start
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "mov %[weights], %[weights]\n\t"
#else
                    "mov %[weights], x2\n\t"
#endif

                    // Prepare the destination base for next 4 batches
                    "mov %[dst_1], %[dst_3]\n\t"
                    "add %[dst_2], %[dst_1], %[rows], asr #2\n\t"
                    "add %[dst_3], %[dst_2], %[rows], asr #2\n\t"
                    "add %[dst_4], %[dst_3], %[rows], asr #2\n\t"

                    // Check if the all the columns of weight matrix are processed
                    "add %w[k], %w[k], #4\n\t"
                    "cmp %w[k], %w[batches]\n\t"
                    "b.lt 5b\n\t"


                    : [ dst_1 ]      "+r" (_output_1),   [ dst_2 ]       "+r" (_output_2),
                      [ dst_3 ]      "+r" (_output_3),   [ dst_4 ]       "+r" (_output_4),
                      [ i ]          "+r" (i),           [ end ]         "+r" (end),
                      [ j ]          "+r" (j),           [ k ]           "+r" (k)

                    : [ activation ] "r"  (_input),      [ act_base ]    "r"  (_input_base),
                      [ weights ]    "r"  (_kernel),     [ wts_base ]    "r"  (_kernel_base),
                      [ size ]       "r"  (lhs_columns), [ rows ]        "r"  (rhs_rows),
                      [ batches ]    "r"  (lhs_batches)

                    : "v0",  "v1",  "v2",  "v3",
                      "v4",  "v5",  "v6",  "v7",
                      "v8",  "v9",  "v10", "v11",
                      "v12", "v13", "v14", "v15",
                      "v16", "v17", "v18", "v19",
                      "v20", "v21", "v22", "v23",
                      "v24", "v25", "v26", "v27",
                      "v28", "v29", "v30", "v31",
                      "x0" , "x1" , "x2"
                );
#else
                int32_t*        _output_1   = output + 0 * rhs_rows;
                int32_t*        _output_2   = output + 1 * rhs_rows;
                /* Vector assignments:
                    * W, WH     -> v0-3      (Weights, Weights High)
                    * A, AH     -> v4-5      (Activations, Activations High)
                    * AL        -> v6-7      (Activations Low)
                    * WL        -> v8-11     (Activations Low)
                    * MiniACC   -> v12-15    (Weights Low)
                    *  -        -> v16-19    (Not Used)
                    *  -        -> v20-23    (Not Used)
                    * ACC1      -> v24-27    (Accumulators input row #1)
                    * ACC2      -> v28-31    (Accumulators input row #2)
                */
                asm volatile(
                    "mov %w[k], wzr\n\t"

                    // Start of The Loop Over Batches
                    "5:\n\t"
                    "mov %w[j], wzr\n\t"

                    "0:\n\t"
                    "mov %w[i], wzr\n\t"
                    "movi v16.4s, #0\n\t"
                    "movi v17.4s, #0\n\t"
                    "movi v18.4s, #0\n\t"
                    "movi v19.4s, #0\n\t"
                    "movi v20.4s, #0\n\t"
                    "movi v21.4s, #0\n\t"
                    "movi v22.4s, #0\n\t"
                    "movi v23.4s, #0\n\t"
                    "movi v24.4s, #0\n\t"
                    "movi v25.4s, #0\n\t"
                    "movi v26.4s, #0\n\t"
                    "movi v27.4s, #0\n\t"
                    "movi v28.4s, #0\n\t"
                    "movi v29.4s, #0\n\t"
                    "movi v30.4s, #0\n\t"
                    "movi v31.4s, #0\n\t"

                    // Start of Outer Loop Over Weights

                    
                    "1:\n\t"
                    // Load Weights
                    "ld1 {v0.16b},  [%[weights]], #16\n\t"
                    "ld1 {v1.16b},  [%[weights]], #16\n\t"
                    "ld1 {v2.16b},  [%[weights]], #16\n\t"
                    "ld1 {v3.16b},  [%[weights]], #16\n\t"

                    // Load Activations
                    "ld1 {v4.16b},  [%[activation]], #16\n\t"
                    "ld1 {v5.16b},  [%[activation]], #16\n\t"

                    // SHL WL, W, #4
                    "shl v8.16b, v0.16b,  #4\n\t"
                    "shl v9.16b, v1.16b,  #4\n\t"
                    "shl v10.16b, v2.16b, #4\n\t"
                    "shl v11.16b, v3.16b, #4\n\t"
                    
                    // SSHR WL, WL, #4
                    "sshr v8.16b, v8.16b, #4\n\t"
                    "sshr v9.16b, v9.16b, #4\n\t"
                    "sshr v10.16b, v10.16b, #4\n\t"
                    "sshr v11.16b, v11.16b, #4\n\t"

                    // SSHR WH, W, #4
                    "sshr v0.16b, v0.16b, #4\n\t"
                    "sshr v1.16b, v1.16b, #4\n\t"
                    "sshr v2.16b, v2.16b, #4\n\t"
                    "sshr v3.16b, v3.16b, #4\n\t"
                    
                    // SHL AL, A, #4
                    "shl v6.16b,  v4.16b,  #4\n\t"
                    "shl v7.16b,  v5.16b,  #4\n\t"

                    // SSHR AH, A, #4
                    "sshr v4.16b, v4.16b, #4\n\t"
                    "sshr v5.16b, v5.16b, #4\n\t"

                    // SSHR AL, AL, #4
                    "sshr v6.16b,  v6.16b,  #4\n\t"
                    "sshr v7.16b,  v7.16b,  #4\n\t"

                    // Activation Row #1
                    // SMULL MiniACC, WL, AL[0]
                    "smull v12.8h, v0.8b, v6.8b\n\t"
                    "smull v13.8h, v1.8b, v6.8b\n\t"
                    "smull v14.8h, v2.8b, v6.8b\n\t"
                    "smull v15.8h, v3.8b, v6.8b\n\t"

                    // SMLAL2 MiniACC, WL, AL[0]
                    "smlal2 v12.8h, v0.16b, v6.16b\n\t"
                    "smlal2 v13.8h, v1.16b, v6.16b\n\t"
                    "smlal2 v14.8h, v2.16b, v6.16b\n\t"
                    "smlal2 v15.8h, v3.16b, v6.16b\n\t"

                    // SADALP ACC1, MiniACC
                    "sadalp v24.4s, v12.8h\n\t"
                    "sadalp v25.4s, v13.8h\n\t"
                    "sadalp v26.4s, v14.8h\n\t"
                    "sadalp v27.4s, v15.8h\n\t"

                    // Activation Row #2
                    // SMULL MiniACC, WL, AL[1]
                    "smull v12.8h, v0.8b, v7.8b\n\t"
                    "smull v13.8h, v1.8b, v7.8b\n\t"
                    "smull v14.8h, v2.8b, v7.8b\n\t"
                    "smull v15.8h, v3.8b, v7.8b\n\t"

                    // SMLAL2 MiniACC, WL, AL[1]
                    "smlal2 v12.8h, v0.16b, v7.16b\n\t"
                    "smlal2 v13.8h, v1.16b, v7.16b\n\t"
                    "smlal2 v14.8h, v2.16b, v7.16b\n\t"
                    "smlal2 v15.8h, v3.16b, v7.16b\n\t"

                    // SADALP ACC2, MiniACC
                    "sadalp v28.4s, v12.8h\n\t"
                    "sadalp v29.4s, v13.8h\n\t"
                    "sadalp v30.4s, v14.8h\n\t"
                    "sadalp v31.4s, v15.8h\n\t"

                    // 
                    // Higher half of weight vectors
                    // 

                    // Activation Row #1
                    // SMULL MiniACC, WH, AH[0]
                    "smull v12.8h, v8.8b,  v4.8b\n\t"
                    "smull v13.8h, v9.8b,  v4.8b\n\t"
                    "smull v14.8h, v10.8b, v4.8b\n\t"
                    "smull v15.8h, v11.8b, v4.8b\n\t"

                    // SMLAL2 MiniACC, WH, AH[0]
                    "smlal2 v12.8h, v8.16b,  v4.16b\n\t"
                    "smlal2 v13.8h, v9.16b,  v4.16b\n\t"
                    "smlal2 v14.8h, v10.16b, v4.16b\n\t"
                    "smlal2 v15.8h, v11.16b, v4.16b\n\t"

                    // SADALP ACC1, MiniACC
                    "sadalp v24.4s, v12.8h\n\t"
                    "sadalp v25.4s, v13.8h\n\t"
                    "sadalp v26.4s, v14.8h\n\t"
                    "sadalp v27.4s, v15.8h\n\t"

                    // Activation Row #2
                    // SMULL MiniACC, WH, AH[1]
                    "smull v12.8h, v8.8b,  v5.8b\n\t"
                    "smull v13.8h, v9.8b,  v5.8b\n\t"
                    "smull v14.8h, v10.8b, v5.8b\n\t"
                    "smull v15.8h, v11.8b, v5.8b\n\t"

                    // SMLAL2 MiniACC, WH, AH[1]
                    "smlal2 v12.8h, v8.16b,  v5.16b\n\t"
                    "smlal2 v13.8h, v9.16b,  v5.16b\n\t"
                    "smlal2 v14.8h, v10.16b, v5.16b\n\t"
                    "smlal2 v15.8h, v11.16b, v5.16b\n\t"

                    // SADALP ACC2, MiniACC
                    "sadalp v28.4s, v12.8h\n\t"
                    "sadalp v29.4s, v13.8h\n\t"
                    "sadalp v30.4s, v14.8h\n\t"
                    "sadalp v31.4s, v15.8h\n\t"

                    // Check if the loop over rows of weight matrix is done
                    "add %w[i], %w[i], #32\n\t"
                    "cmp %w[i], %w[size]\n\t"
                    "b.lt 1b\n\t"

                    // Accumulate the ACC1 to one int32
                    "addv s24, v24.4s\n\t"
                    "addv s25, v25.4s\n\t"
                    "addv s26, v26.4s\n\t"
                    "addv s27, v27.4s\n\t"

                    // Accumulate the ACC2 to one int32
                    "addv s28, v28.4s\n\t"
                    "addv s29, v29.4s\n\t"
                    "addv s30, v30.4s\n\t"
                    "addv s31, v31.4s\n\t"

                    // Reorder ACC1 to store
                    "mov v24.s[1], v25.s[0]\n\t"
                    "mov v24.s[2], v26.s[0]\n\t"
                    "mov v24.s[3], v27.s[0]\n\t"

                    // Reorder ACC2 to store
                    "mov v28.s[1], v29.s[0]\n\t"
                    "mov v28.s[2], v30.s[0]\n\t"
                    "mov v28.s[3], v31.s[0]\n\t"
                    
                    // Store the 4 int32 results
                    "st1 {v24.4s},  [%[dst_1]], #16\n\t"
                    "st1 {v28.4s},  [%[dst_2]], #16\n\t"
                    
                    // Reset the activations to the start of the row
                    "mov %[activation], %[act_base]\n\t"

                    // Check if the all the columns of weight matrix are processed
                    "add %w[j], %w[j], #4\n\t"
                    "cmp %w[j], %w[rows]\n\t"
                    "b.lt 0b\n\t"

                    // Prepare the activation base for next 4 batches
                    "add %[act_base], %[act_base], %[size], asr #2\n\t"
                    
                    // Reset the activations to the start of the row
                    "mov %[activation], %[act_base]\n\t"

                    // Reset the weights to the start
                    "mov %[weights], %[wts_base]\n\t"

                    // Prepare the destination base for next 2 batches
                    "mov %[dst_1], %[dst_2]\n\t"
                    "add %[dst_2], %[dst_1], %[rows], asr #2\n\t"

                    // Check if the all the columns of weight matrix are processed
                    "add %w[k], %w[k], #2\n\t"
                    "cmp %w[k], %w[batches]\n\t"
                    "b.lt 5b\n\t"


                    : [ dst_1 ]      "+r" (_output_1),   [ dst_2 ]       "+r" (_output_2),
                      [ i ]          "+r" (i),           [ end ]         "+r" (end),
                      [ j ]          "+r" (j),           [ k ]           "+r" (k)

                    : [ activation ] "r"  (_input),      [ act_base ]    "r"  (_input_base),
                      [ weights ]    "r"  (_kernel),     [ wts_base ]    "r"  (_kernel_base),
                      [ size ]       "r"  (lhs_columns), [ rows ]        "r"  (rhs_rows),
                      [ batches ]    "r"  (lhs_batches)

                    : "v0",  "v1",  "v2",  "v3",
                      "v4",  "v5",  "v6",  "v7",
                      "v8",  "v9",  "v10", "v11",
                      "v12", "v13", "v14", "v15",
                      "v16", "v17", "v18", "v19",
                      "v20", "v21", "v22", "v23",
                      "v24", "v25", "v26", "v27",
                      "v28", "v29", "v30", "v31"
                );
#endif
#endif
#endif
                return Status::Success;
            }
            Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const Params params
            ){}
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
                    "mov %w[i], wzr\n\t"
                    "dup v23.4s, wzr\n\t"
                    "dup v24.4s, wzr\n\t"
                    "dup v25.4s, wzr\n\t"
                    "dup v30.4s, wzr\n\t"
                    
                    "cmp %w[size], #0\n\t"
                    "beq 3f\n\t"

                    // Start of Outer Loop Over Weights
                    "1:\n\t"
                    "ld1 {v0.16b},  [%[weights]], #16\n\t"
                    "ld1 {v1.16b},  [%[weights]], #16\n\t"
                    "ld1 {v2.16b},  [%[weights]], #16\n\t"
                    "ld1 {v3.16b},  [%[weights]], #16\n\t"

                    "ld1 {v4.16b},  [%[activation]], #16\n\t"

                    // SHL AL, A, #4
                    "shl v5.16b,  v4.16b,  #4\n\t"
                    // SSHR AH, A, #4
                    "sshr v4.16b, v4.16b, #4\n\t"
                    // SSHR AL, AL, #4
                    "sshr v5.16b,  v5.16b,  #4\n\t"
                    
                    // Load next weights
                    "ld1 {v8.16b},  [%[weights]], #16\n\t"

                    // SMLAL T1.8h, WH.8b, A2.8b
                    "smlal v13.8h, v0.8b, v5.8b\n\t"
                    "smlal v14.8h, v1.8b, v5.8b\n\t"
                    "smlal v15.8h, v2.8b, v5.8b\n\t"
                    "smlal v16.8h, v3.8b, v5.8b\n\t"
                    
                    // Load more rows of next weights
                    "ld1 {v10.16b},  [%[weights]], #16\n\t"

                    // SMLAL2 T1.8h, WH.16b, A2.16b
                    "smlal2 v13.8h, v0.16b, v5.16b\n\t"
                    "smlal2 v14.8h, v1.16b, v5.16b\n\t"
                    "smlal2 v15.8h, v2.16b, v5.16b\n\t"
                    "smlal2 v16.8h, v3.16b, v5.16b\n\t"

                    // Load more rows of next weights
                    "ld1 {v11.16b},  [%[weights]], #16\n\t"
                    "ld1 {v12.16b},  [%[weights]], #16\n\t"

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
                /* const int8_t* _activation = activation;
                int i, end;
                int8_t mask = 0x0F;
                // Vector assignments:
                //  * W    -> v0-3
                //  * A    -> v4-7
                //  * CW   -> v8-11
                //  * T    -> v12-15
                //  * ACC1 -> v16-19
                //  * ACC2 -> v20-23
                //  * ACC3 -> v24-27
                //  * ACC4 -> v28-31
                
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
                );*/
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
    }
}
#endif

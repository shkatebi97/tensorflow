#ifdef BAZEL_BUILD
#include "low_precision_fully_connected.h"
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
        namespace BinaryInputsInt8Weights {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape){
                int padding_size = (shape.size[1] % 16)?(16 - (shape.size[1] % 16)):(0);
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
                shape[n_dims - 1] = ceil(shape[n_dims - 1] / 16.0) * 16 / (8 / 8);
                return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
            }
            size_t TransformInputShape(int* shape, int n_dims){
                shape[n_dims - 1] = ::ceil(shape[n_dims - 1] / 16.0) * 16 / (8 / 1);
                return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
            }
            Status QuantizeFilter(const int8_t* input, Shape k_shape, int8_t* output, MemLayout layout){
                if (k_shape.number_dims != 2)
                    return Status::DimensionsMisMatch;
                if (k_shape.size[1] % 128)
                    return Status::SizesMisMatch;
                if (k_shape.size[0] % 4)
                    return Status::SizesMisMatch;
                if (layout != MemLayout::kRowMajor)
                    return Status::WrongMemLayout;
                doLowPrecisionWeightPack(const_cast<int8_t*>(input), output, k_shape.size[0], k_shape.size[1]);
                return Status::Success;
            }
            Status QuantizeInput(const int8_t* input, Shape shape, int8_t* output, MemLayout layout){
                bool is_multibatched = shape.number_dims == 2 && shape.size[0] > 1;
                if (is_multibatched)
                    return Status::NotImplemented; 
                if (!((shape.number_dims == 2 && shape.size[0] == 1) || shape.number_dims == 1))
                    return Status::DimensionsMisMatch;
                if (shape.size[shape.number_dims - 1] % 128)
                    return Status::SizesMisMatch; 
                if (layout != MemLayout::kRowMajor)
                    return Status::WrongMemLayout;
                if (GetVariableFromEnv("DismissInputQuantization") == std::string("TRUE") ||
                    GetVariableFromEnv("DismissQuantization") == std::string("TRUE")){
                    std::copy(input, input + (shape.flatsize / 8), output);
                }
                else {
                    int i , size = shape.flatsize;
                    asm volatile(
                        "mov %w[i], wzr\n\t"
                        "movi v30.16b, #7\n\t"
                        "movi v29.16b, #6\n\t"
                        "movi v28.16b, #5\n\t"
                        "movi v27.16b, #4\n\t"
                        "movi v26.16b, #3\n\t"
                        "movi v25.16b, #2\n\t"
                        "movi v24.16b, #1\n\t"
                        
                        "cmp %w[size], #0\n\t"
                        "beq 3f\n\t"

                        // Start of Outer Loop Over Weights
                        "1:\n\t"
                        // Load A1
                        "ld1 {v0.16b},  [%[input]], #16\n\t"
                        "ld1 {v1.16b},  [%[input]], #16\n\t"
                        "ld1 {v2.16b},  [%[input]], #16\n\t"
                        "ld1 {v3.16b},  [%[input]], #16\n\t"

                        // Get Sign

                        // USHR A1T, A1, 7
                        "ushr v0.16b, v0.16b, #7\n\t"
                        "ushr v1.16b, v1.16b, #7\n\t"
                        "ushr v2.16b, v2.16b, #7\n\t"
                        "ushr v3.16b, v3.16b, #7\n\t"

                        // Load A2
                        "ld1 {v4.16b},  [%[input]], #16\n\t"
                        "ld1 {v5.16b},  [%[input]], #16\n\t"
                        "ld1 {v6.16b},  [%[input]], #16\n\t"
                        "ld1 {v7.16b},  [%[input]], #16\n\t"

                        // Get Sign

                        // USHR A2T, A2, 7
                        "ushr v4.16b, v4.16b, #7\n\t"
                        "ushr v5.16b, v5.16b, #7\n\t"
                        "ushr v6.16b, v6.16b, #7\n\t"
                        "ushr v7.16b, v7.16b, #7\n\t"

                        // USHL A1S, A1S, x
                        "ushl v0.16b, v0.16b, v30.16b\n\t"
                        "ushl v1.16b, v1.16b, v29.16b\n\t"
                        "ushl v2.16b, v2.16b, v28.16b\n\t"
                        "ushl v3.16b, v3.16b, v27.16b\n\t"

                        // USHL A2S, A2S, x
                        "ushl v4.16b, v4.16b, v26.16b\n\t"
                        "ushl v5.16b, v5.16b, v25.16b\n\t"
                        "ushl v6.16b, v6.16b, v24.16b\n\t"

                        // ORR AM, A1S
                        "orr v0.16b, v0.16b, v1.16b\n\t"
                        "orr v0.16b, v0.16b, v2.16b\n\t"
                        "orr v0.16b, v0.16b, v3.16b\n\t"

                        // ORR AM, A2S
                        "orr v0.16b, v0.16b, v4.16b\n\t"
                        "orr v0.16b, v0.16b, v5.16b\n\t"
                        "orr v0.16b, v0.16b, v6.16b\n\t"
                        "orr v0.16b, v0.16b, v7.16b\n\t"
                        
                        // ST1 AM, output
                        "st1 {v0.16b},  [%[output]], #16\n\t"

                        "add %w[i], %w[i], #128\n\t"
                        "cmp %w[i], %w[size]\n\t"
                        "b.lt 1b\n\t"

                        "sub %[input], %[input], %[size]\n\t"

                        "3:\n\t"

                        : [ output ] "+r"(output), [ i ] "+r"(i)
                        : [ input ]  "r" (input), [ size ] "r"(size)
                        : "v0",  "v1",  "v2",  "v3",
                          "v4",  "v5",  "v6",  "v7",
                          "v28", "v29", "v30", "v31",
                          "w3",  "w4",  "w5",  "w6"
                    );
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
                int i, j, end;

                asm volatile(
                    "mov %w[j], wzr\n\t"
                    "movi v29.16b, #1\n\t"

                    "0:\n\t"
                    "mov %w[i], wzr\n\t"
                    
                    // Reseting AC
                    "dup v23.4s, wzr\n\t"
                    "dup v24.4s, wzr\n\t"
                    "dup v25.4s, wzr\n\t"
                    "dup v30.4s, wzr\n\t"
                    
                    // If size is zero, discard
                    "cmp %w[size], #0\n\t"
                    "beq 3f\n\t"

                    // Start of Outer Loop Over Weights
                    "1:\n\t"
                    "ld1 {v4.16b},  [%[activation]], #16\n\t"

                    // Setting the iterations of inner loop
                    "add %w[end], %w[i], #128\n\t"
                    "tst %w[size], %w[end]\n\t"
                    "csel %w[end], %w[end], %w[size], lo\n\t"

                    // Reseting MiniAC
                    "dup v26.8h, wzr\n\t"
                    "dup v27.8h, wzr\n\t"
                    "dup v28.8h, wzr\n\t"
                    "dup v31.8h, wzr\n\t"
                    
                    // Start of Inner Loop Over Activations
                    "2:\n\t"

                    // Loading Next Weights
                    "ld1 {v0.16b},  [%[weights]], #16\n\t"
                    "ld1 {v1.16b},  [%[weights]], #16\n\t"
                    "ld1 {v2.16b},  [%[weights]], #16\n\t"
                    "ld1 {v3.16b},  [%[weights]], #16\n\t"
                    
                    // SSHR AT, A, #7
                    "sshr v5.16b,  v4.16b,  #7\n\t"

                    // ORR AT, AT, 1
                    "orr v5.16b,  v5.16b,  v29.16b\n\t"
                    
                    // SMLAL2 MiniAC.8h, W.8b, AT.8b
                    "smlal v26.8h, v0.8b, v5.8b\n\t"
                    "smlal v27.8h, v1.8b, v5.8b\n\t"
                    "smlal v28.8h, v2.8b, v5.8b\n\t"
                    "smlal v31.8h, v3.8b, v5.8b\n\t"

                    // SMLAL2 MiniAC.8h, W.16b, AT.16b
                    "smlal2 v26.8h, v0.16b,  v5.16b\n\t"
                    "smlal2 v27.8h, v1.16b, v5.16b\n\t"
                    "smlal2 v28.8h, v2.16b, v5.16b\n\t"
                    "smlal2 v31.8h, v3.16b, v5.16b\n\t"
                    
                    // SHL A, A , #1
                    "shl v4.16b, v4.16b, #1\n"

                    // Increment the loop counter with 16 and compare with end
                    "add %w[i], %w[i], #16\n\t"
                    "cmp %w[i], %w[end]\n\t"
                    "b.lt 2b\n\t"

                    // ACCUMULATE ACC, MiniAC
                    "sadalp v23.4s, v26.8h\n\t"
                    "sadalp v24.4s, v27.8h\n\t"
                    "sadalp v25.4s, v28.8h\n\t"
                    "sadalp v30.4s, v31.8h\n\t"

                    // Check if the whole row is processed
                    "cmp %w[i], %w[size]\n\t"
                    "b.lt 1b\n\t"

                    // Add the 4 values inisde each vector to each other
                    "addv s23, v23.4s\n\t"
                    "addv s24, v24.4s\n\t"
                    "addv s25, v25.4s\n\t"
                    "addv s30, v30.4s\n\t"

                    // Put each rows end result, inside a vector
                    "mov v23.s[1], v24.s[0]\n\t"
                    "mov v23.s[2], v25.s[0]\n\t"
                    "mov v23.s[3], v30.s[0]\n\t"

                    // Save the end result for 4 rows
                    "st1 {v23.4s},  [%[dst]], #16\n\t"

                    // Reseting activation to start
                    "mov %[activation], %[lhs_base]\n\t"

                    "3:\n\t"

                    // Check if whole matrix is processed and we are done 
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

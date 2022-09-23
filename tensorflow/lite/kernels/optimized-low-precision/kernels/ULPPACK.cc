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
        namespace ULPPACK {
            size_t TransformFilterShape(int* shape, int n_dims){
                shape[n_dims - 1] = ceil(shape[n_dims - 1] / 16.0) * 16 / (8 / 8);
                return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
            }
            size_t TransformInputShape(int* shape, int n_dims){
                shape[n_dims - 1] = ::ceil(shape[n_dims - 1] / 16.0) * 16 / (8 / 8);
                return ::LowPrecision::FullyConnected::CalcFlatSize(shape, n_dims);
            }
            Status QuantizeFilter(const int8_t* input, Shape k_shape, int8_t* output, MemLayout layout){
                if (k_shape.number_dims != 2)
                    return Status::DimensionsMisMatch;
                if (k_shape.size[1] % 16)
                    return Status::SizesMisMatch;
                if (k_shape.size[0] % 4)
                    return Status::SizesMisMatch;
                if (layout != MemLayout::kRowMajor)
                    return Status::WrongMemLayout;
                doLowPrecisionWeightPack(const_cast<int8_t*>(input), output, k_shape.size[0], k_shape.size[1]);
                return Status::Success;
            }
            Status QuantizeInput(const int8_t* input, Shape shape, int8_t* output, MemLayout layout){
                if (shape.size[shape.number_dims - 1] % 16)
                    return Status::SizesMisMatch; 
                if (layout != MemLayout::kRowMajor)
                    return Status::WrongMemLayout;
                bool is_multibatched = shape.number_dims == 2 && shape.size[0] > 1;
                if (is_multibatched && shape.size[0] % 4)
                    return Status::SizesMisMatch; 
                if (GetVariableFromEnv("DismissInputQuantization") == std::string("TRUE") ||
                    GetVariableFromEnv("DismissQuantization") == std::string("TRUE")){
                    if (is_multibatched)
                        std::copy(input, input + shape.flatsize, output);
                    else
                        return Status::NotNeeded;
                }
                else {
                    if (is_multibatched){
                        int8_t* input_casted = const_cast<int8_t*>(input);
                        doLowPrecisionWeightPack(input_casted, output, shape.size[0], shape.size[1]);
                    }
                    else
                        return Status::NotNeeded;
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
                
                // int8_t*         _kernel = const_cast<int8_t*>(kernel);
                // int32_t*        _output = output;
                // int i, j, end;

                // // Matrix a : 4 x k
                // // Matrix w : k x 8
                // // perform matrix multiply a x w (Result Matrix: 4x8)
                // // assume both a and w are multipacked by depth-2.
                // // Kernel for depth-2, local accumulation-4 
                // // Overflow safe below W3A2 (See Figure8 in the paper)
                // static void ukernel_4x8__neon_multipack_depth2_localacc4(
                //     size_t mr,
                //     size_t nr,
                //     size_t k,
                //     const uint8_t* __restrict__ a,
                //     size_t a_stride,
                //     const uint8_t* __restrict__ w,
                //     int32_t* __restrict__ c,
                //     size_t c_stride) {
                int i = 0, k = lhs_columns;
                for (; rhs_rows >= 8; rhs_rows -= 8) {
                    const uint8_t* a0 = (uint8_t*)input;
                    const uint8_t* w  = (uint8_t*)(kernel + (i * (k * 8 * 16)));

                    // Initializing result
                    uint16x8_t vacc0;
                    uint16x8_t vacc0x = veorq_u16(vacc0x, vacc0x);
                    
                    // Decrementing 16 : Depth(2) * LocalAcc(4) * Unroll(2)
                    // Note that you need to handle final corner cases (k<=16), 
                    // but I just omitted in this snippet.
                    for (; k >= 16; k -= 16) {
                        uint16x4_t vxa0 = vld1_u16((uint16_t*)a0); a0 += 8;

                        uint16x8_t vxb01234567c0 = vld1q_u16((uint16_t*)w); w += 16;
                        uint16x8_t vxb01234567c1 = vld1q_u16((uint16_t*)w); w += 16;
                        uint16x8_t vxb01234567c2 = vld1q_u16((uint16_t*)w); w += 16;
                        uint16x8_t vxb01234567c3 = vld1q_u16((uint16_t*)w); w += 16;
                        
                        // First local accumulation
                        vacc0 = vmulq_lane_u16(vxb01234567c0, vxa0, 0);
                        
                        // Second local accumulation
                        vacc0 = vmlaq_lane_u16(vacc0, vxb01234567c1, vxa0, 1);
                            
                        // Third local accumulation
                        vacc0 = vmlaq_lane_u16(vacc0, vxb01234567c2, vxa0, 2);
                        
                        // Fourth local accumulation
                        vacc0 = vmlaq_lane_u16(vacc0, vxb01234567c3, vxa0, 3);
                        
                        vacc0x = vsraq_n_u16(vacc0x, vacc0, 8);

                        // UNROLLED
                        vxa0 = vld1_u16((uint16_t*)a0); a0 += 8;

                        vxb01234567c0 = vld1q_u16((uint16_t*)w); w += 16;
                        vxb01234567c1 = vld1q_u16((uint16_t*)w); w += 16;
                        vxb01234567c2 = vld1q_u16((uint16_t*)w); w += 16;
                        vxb01234567c3 = vld1q_u16((uint16_t*)w); w += 16;

                        vacc0 = vmulq_lane_u16(vxb01234567c0, vxa0, 0);

                        vacc0 = vmlaq_lane_u16(vacc0, vxb01234567c1, vxa0, 1);

                        vacc0 = vmlaq_lane_u16(vacc0, vxb01234567c2, vxa0, 2);
                        
                        vacc0 = vmlaq_lane_u16(vacc0, vxb01234567c3, vxa0, 3);
                        
                        vacc0x = vsraq_n_u16(vacc0x, vacc0, 8);
                    }
                    
                    uint32x4_t vacc0x0123_; 
                    uint32x4_t vacc0x4567_; 
                    
                    vacc0x0123_ = vmovl_u16(vget_low_u16(vacc0x));
                    vacc0x4567_ = vmovl_u16(vget_high_u16(vacc0x));
                    
                    int32x4_t vacc0x0123 = vreinterpretq_s32_u32(vacc0x0123_);
                    int32x4_t vacc0x4567 = vreinterpretq_s32_u32(vacc0x4567_);

                    // Storing result
                    int32_t* c0 = output + (i * 8);
                    
                    vst1q_s32(c0, vaddq_s32(vld1q_s32(c0),vacc0x0123));
                    vst1q_s32(c0+4, vaddq_s32(vld1q_s32(c0+4),vacc0x4567));

                    i += 8;
                }
                return Status::Success;
            }
            // Status MultiplyInt8SingleBatch(
            //     const int8_t* input, Shape input_shape,
            //     const int8_t* kernel, Shape kernel_shape,
            //     int32_t* output, Shape output_shape
            // ){
            //     int lhs_columns = input_shape.size[input_shape.number_dims - 1] ,
            //         rhs_rows = kernel_shape.size[0] ,
            //         rhs_columns = kernel_shape.size[1];
            //     if (lhs_columns != rhs_columns)
            //         return Status::SizesMisMatch;
            //     if(lhs_columns == 0 || rhs_rows == 0)
            //         return Status::Success;
            //                
            //     int8_t*         _kernel = const_cast<int8_t*>(kernel);
            //     const int8_t*   _input  = input;
            //     int32_t*        _output = output;
            //     int i, j, end;
            //
            //     asm volatile(
            //         "mov %w[j], wzr\n\t"
            //         "movi v29.16b, #1\n\t"
            //
            //         "0:\n\t"
            //         "mov %w[i], wzr\n\t"
            //                    
            //         // Reseting AC
            //         "dup v23.4s, wzr\n\t"
            //         "dup v24.4s, wzr\n\t"
            //         "dup v25.4s, wzr\n\t"
            //         "dup v30.4s, wzr\n\t"
            //                    
            //         // If size is zero, discard
            //         "cmp %w[size], #0\n\t"
            //         "beq 3f\n\t"
            //
            //         // Start of Outer Loop Over Weights
            //         "1:\n\t"
            //         "ld1 {v4.16b},  [%[activation]], #16\n\t"
            //
            //         // Setting the iterations of inner loop
            //         "add %w[end], %w[i], #128\n\t"
            //         "tst %w[size], %w[end]\n\t"
            //         "csel %w[end], %w[end], %w[size], lo\n\t"
            //
            //         // Reseting MiniAC
            //         "dup v26.8h, wzr\n\t"
            //         "dup v27.8h, wzr\n\t"
            //         "dup v28.8h, wzr\n\t"
            //         "dup v31.8h, wzr\n\t"
            //                   
            //         // Start of Inner Loop Over Activations
            //         "2:\n\t"
            //
            //         // Loading Next Weights
            //         "ld1 {v0.16b},  [%[weights]], #16\n\t"
            //         "ld1 {v1.16b},  [%[weights]], #16\n\t"
            //         "ld1 {v2.16b},  [%[weights]], #16\n\t"
            //         "ld1 {v3.16b},  [%[weights]], #16\n\t"
            //                    
            //         // SSHR AT, A, #7
            //         "sshr v5.16b,  v4.16b,  #7\n\t"
            //
            //         // ORR AT, AT, 1
            //         "orr v5.16b,  v5.16b,  v29.16b\n\t"
            //                    
            //         // SMLAL2 MiniAC.8h, W.8b, AT.8b
            //         "smlal v26.8h, v0.8b, v5.8b\n\t"
            //         "smlal v27.8h, v1.8b, v5.8b\n\t"
            //         "smlal v28.8h, v2.8b, v5.8b\n\t"
            //         "smlal v31.8h, v3.8b, v5.8b\n\t"
            //
            //         // SMLAL2 MiniAC.8h, W.16b, AT.16b
            //         "smlal2 v26.8h, v0.16b,  v5.16b\n\t"
            //         "smlal2 v27.8h, v1.16b, v5.16b\n\t"
            //         "smlal2 v28.8h, v2.16b, v5.16b\n\t"
            //         "smlal2 v31.8h, v3.16b, v5.16b\n\t"
            //                    
            //         // SHL A, A , #1
            //         "shl v4.16b, v4.16b, #1\n"
            //
            //         // Increment the loop counter with 16 and compare with end
            //         "add %w[i], %w[i], #16\n\t"
            //         "cmp %w[i], %w[end]\n\t"
            //         "b.lt 2b\n\t"
            //
            //         // ACCUMULATE ACC, MiniAC
            //         "sadalp v23.4s, v26.8h\n\t"
            //         "sadalp v24.4s, v27.8h\n\t"
            //         "sadalp v25.4s, v28.8h\n\t"
            //         "sadalp v30.4s, v31.8h\n\t"
            //
            //         // Check if the whole row is processed
            //         "cmp %w[i], %w[size]\n\t"
            //         "b.lt 1b\n\t"
            //
            //         // Add the 4 values inisde each vector to each other
            //         "addv s23, v23.4s\n\t"
            //         "addv s24, v24.4s\n\t"
            //         "addv s25, v25.4s\n\t"
            //         "addv s30, v30.4s\n\t"
            //
            //         // Put each rows end result, inside a vector
            //         "mov v23.s[1], v24.s[0]\n\t"
            //         "mov v23.s[2], v25.s[0]\n\t"
            //         "mov v23.s[3], v30.s[0]\n\t"
            //
            //         // Save the end result for 4 rows
            //         "st1 {v23.4s},  [%[dst]], #16\n\t"
            //
            //         // Reseting activation to start
            //         "mov %[activation], %[lhs_base]\n\t"
            //
            //         "3:\n\t"
            //
            //         // Check if whole matrix is processed and we are done 
            //         "add %w[j], %w[j], #4\n\t"
            //         "cmp %w[j], %w[rows]\n\t"
            //         "b.lt 0b\n\t"
            //
            //
            //         : [ dst ]        "+r" (_output),     [ i ]           "+r" (i),
            //           [ j ]          "+r" (j),           [ end ]         "+r" (end)
            //
            //         : [ activation ] "r"  (_input),      [ lhs_base ]    "r"  (input),
            //           [ weights ]    "r"  (_kernel),     [ size ]        "r"  (lhs_columns),
            //           [ rows ]       "r"  (rhs_rows)
            //
            //         : "v0",  "v1",  "v2",  "v3",
            //           "v4",  "v5",  "v6",  "v7",
            //           "v8",  "v9",  "v10", "v11",
            //           "v12", "v13", "v14", "v15",
            //           "v16", "v17", "v18", "v19",
            //           "v20", "v21", "v22", "v23",
            //           "v24", "v25", "v26", "v27",
            //           "v28", "v29", "v30", "v31"
            //     );
            //
            //     return Status::Success;
            // }
            Status MultiplyInt8MultiBatched(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape
            ){
                int lhs_batches = input_shape.size[0],
                    lhs_columns = input_shape.size[1],
                    rhs_rows    = kernel_shape.size[0],
                    rhs_columns = kernel_shape.size[1];
                
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
                int32_t*        _output_1   = output + 0 * rhs_rows;
                int32_t*        _output_2   = output + 1 * rhs_rows;
                int32_t*        _output_3   = output + 2 * rhs_rows;
                int32_t*        _output_4   = output + 3 * rhs_rows;
                /* Vector assignments:
                    * W         -> v0-3      (Weights)
                    * A         -> v4-7      (Activations)
                    * MW        -> v8-11     (Masked Weights)
                    * MiniACC   -> v12-15    (Mini Accumulator)
                    * ACC1      -> v16-19    (Accumulators input row #1)
                    * ACC2      -> v20-23    (Accumulators input row #2)
                    * ACC3      -> v24-27    (Accumulators input row #3)
                    * ACC4      -> v28-31    (Accumulators input row #4)
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
                    "ld1 {v0.16b},  [%[weights]]\n\t"
                    "ld1 {v1.16b},  [%[weights]]\n\t"
                    "ld1 {v2.16b},  [%[weights]]\n\t"
                    "ld1 {v3.16b},  [%[weights]]\n\t"
#else
                    "ld1 {v0.16b},  [%[weights]], #16\n\t"
                    "ld1 {v1.16b},  [%[weights]], #16\n\t"
                    "ld1 {v2.16b},  [%[weights]], #16\n\t"
                    "ld1 {v3.16b},  [%[weights]], #16\n\t"
#endif

                    // Load Activations
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "ld1 {v4.16b},  [%[activation]]\n\t"
                    "ld1 {v5.16b},  [%[activation]]\n\t"
                    "ld1 {v6.16b},  [%[activation]]\n\t"
                    "ld1 {v7.16b},  [%[activation]]\n\t"
#else
                    "ld1 {v4.16b},  [%[activation]], #16\n\t"
                    "ld1 {v5.16b},  [%[activation]], #16\n\t"
                    "ld1 {v6.16b},  [%[activation]], #16\n\t"
                    "ld1 {v7.16b},  [%[activation]], #16\n\t"
#endif

                    "add %w[end], %w[i], #128\n\t"
                    "tst %w[size], %w[end]\n\t"
                    "csel %w[end], %w[end], %w[size], lo\n\t"
                    
                    // Start of Inner Loop Over Activations
                    "2:\n\t"

                    // SSHR MW, W, #7
                    "sshr v8.16b,  v0.16b, #7\n\t"
                    "sshr v9.16b,  v1.16b, #7\n\t"
                    "sshr v10.16b, v2.16b, #7\n\t"
                    "sshr v11.16b, v3.16b, #7\n\t"

                    // ORR MW, MW, #1
                    "movi v12.16b, 0x01\n\t"
                    "orr v8.16b,  v8.16b,  v12.16b\n\t"
                    "orr v9.16b,  v9.16b,  v12.16b\n\t"
                    "orr v10.16b, v10.16b, v12.16b\n\t"
                    "orr v11.16b, v11.16b, v12.16b\n\t"
                    
                    // Activation Row #1
                    // SMULL MiniACC, MW, A
                    "smull v12.8h, v8.8b,  v4.8b\n\t"
                    "smull v13.8h, v9.8b,  v4.8b\n\t"
                    "smull v14.8h, v10.8b, v4.8b\n\t"
                    "smull v15.8h, v11.8b, v4.8b\n\t"

                    // SMLAL2 MiniACC, MW, A
                    "smlal2 v12.8h, v8.16b,  v4.16b\n\t"
                    "smlal2 v13.8h, v9.16b,  v4.16b\n\t"
                    "smlal2 v14.8h, v10.16b, v4.16b\n\t"
                    "smlal2 v15.8h, v11.16b, v4.16b\n\t"

                    // Load Activations
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "ld1 {v4.16b},  [%[activation]]\n\t"
#else
                    "ld1 {v4.16b},  [%[activation]], #16\n\t"
#endif

                    // SADALP ACC1, MiniACC
                    "sadalp v16.4s, v12.8h\n\t"
                    "sadalp v17.4s, v13.8h\n\t"
                    "sadalp v18.4s, v14.8h\n\t"
                    "sadalp v19.4s, v15.8h\n\t"

                    // Activation Row #2
                    // SMULL MiniACC, MW, A
                    "smull v12.8h, v8.8b,  v5.8b\n\t"
                    "smull v13.8h, v9.8b,  v5.8b\n\t"
                    "smull v14.8h, v10.8b, v5.8b\n\t"
                    "smull v15.8h, v11.8b, v5.8b\n\t"

                    // SMLAL2 MiniACC, MW, A
                    "smlal2 v12.8h, v8.16b,  v5.16b\n\t"
                    "smlal2 v13.8h, v9.16b,  v5.16b\n\t"
                    "smlal2 v14.8h, v10.16b, v5.16b\n\t"
                    "smlal2 v15.8h, v11.16b, v5.16b\n\t"

                    // Load Activations
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "ld1 {v5.16b},  [%[activation]]\n\t"
#else
                    "ld1 {v5.16b},  [%[activation]], #16\n\t"
#endif

                    // SADALP ACC2, MiniACC
                    "sadalp v20.4s, v12.8h\n\t"
                    "sadalp v21.4s, v13.8h\n\t"
                    "sadalp v22.4s, v14.8h\n\t"
                    "sadalp v23.4s, v15.8h\n\t"

                    // Activation Row #3
                    // SMULL MiniACC, MW, A
                    "smull v12.8h, v8.8b,  v6.8b\n\t"
                    "smull v13.8h, v9.8b,  v6.8b\n\t"
                    "smull v14.8h, v10.8b, v6.8b\n\t"
                    "smull v15.8h, v11.8b, v6.8b\n\t"

                    // SMLAL2 MiniACC, MW, A
                    "smlal2 v12.8h, v8.16b,  v6.16b\n\t"
                    "smlal2 v13.8h, v9.16b,  v6.16b\n\t"
                    "smlal2 v14.8h, v10.16b, v6.16b\n\t"
                    "smlal2 v15.8h, v11.16b, v6.16b\n\t"

                    // Load Activations
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "ld1 {v6.16b},  [%[activation]]\n\t"
#else
                    "ld1 {v6.16b},  [%[activation]], #16\n\t"
#endif

                    // SADALP ACC3, MiniACC
                    "sadalp v24.4s, v12.8h\n\t"
                    "sadalp v25.4s, v13.8h\n\t"
                    "sadalp v26.4s, v14.8h\n\t"
                    "sadalp v27.4s, v15.8h\n\t"

                    // Activation Row #4
                    // SMULL MiniACC, MW, A
                    "smull v12.8h, v8.8b,  v7.8b\n\t"
                    "smull v13.8h, v9.8b,  v7.8b\n\t"
                    "smull v14.8h, v10.8b, v7.8b\n\t"
                    "smull v15.8h, v11.8b, v7.8b\n\t"

                    // SMLAL2 MiniACC, MW, A
                    "smlal2 v12.8h, v8.16b,  v7.16b\n\t"
                    "smlal2 v13.8h, v9.16b,  v7.16b\n\t"
                    "smlal2 v14.8h, v10.16b, v7.16b\n\t"
                    "smlal2 v15.8h, v11.16b, v7.16b\n\t"

                    // Load Activations
#ifdef DISABLE_KERNELS_MEM_ACCESS
                    "ld1 {v7.16b},  [%[activation]]\n\t"
#else
                    "ld1 {v7.16b},  [%[activation]], #16\n\t"
#endif

                    // SADALP ACC4, MiniACC
                    "sadalp v28.4s, v12.8h\n\t"
                    "sadalp v29.4s, v13.8h\n\t"
                    "sadalp v30.4s, v14.8h\n\t"
                    "sadalp v31.4s, v15.8h\n\t"

                    // SHL W, W, 7
                    "shl v0.16b, v0.16b, #1\n\t"
                    "shl v1.16b, v1.16b, #1\n\t"
                    "shl v2.16b, v2.16b, #1\n\t"
                    "shl v3.16b, v3.16b, #1\n\t"

                    // Check if the 8 iterations of inner loop are done
                    "add %w[i], %w[i], #16\n\t"
                    "cmp %w[i], %w[end]\n\t"
                    "b.lt 2b\n\t"

                    // Check if the loop over rows of weight matrix (outer loop) is done
                    "cmp %w[i], %w[size]\n\t"
                    "b.lt 1b\n\t"

                    // Accumulate the ACC1 to one int32
                    "addv s16, v16.4s\n\t"
                    "addv s17, v17.4s\n\t"
                    "addv s18, v18.4s\n\t"
                    "addv s19, v19.4s\n\t"

                    // Accumulate the ACC2 to one int32
                    "addv s20, v20.4s\n\t"
                    "addv s21, v21.4s\n\t"
                    "addv s22, v22.4s\n\t"
                    "addv s23, v23.4s\n\t"

                    // Accumulate the ACC3 to one int32
                    "addv s24, v24.4s\n\t"
                    "addv s25, v25.4s\n\t"
                    "addv s26, v26.4s\n\t"
                    "addv s27, v27.4s\n\t"

                    // Accumulate the ACC4 to one int32
                    "addv s28, v28.4s\n\t"
                    "addv s29, v29.4s\n\t"
                    "addv s30, v30.4s\n\t"
                    "addv s31, v31.4s\n\t"

                    // Reorder ACC1 to store
                    "mov v16.s[1], v17.s[0]\n\t"
                    "mov v16.s[2], v18.s[0]\n\t"
                    "mov v16.s[3], v19.s[0]\n\t"

                    // Reorder ACC2 to store
                    "mov v20.s[1], v21.s[0]\n\t"
                    "mov v20.s[2], v22.s[0]\n\t"
                    "mov v20.s[3], v23.s[0]\n\t"

                    // Reorder ACC3 to store
                    "mov v24.s[1], v25.s[0]\n\t"
                    "mov v24.s[2], v26.s[0]\n\t"
                    "mov v24.s[3], v27.s[0]\n\t"

                    // Reorder ACC4 to store
                    "mov v28.s[1], v29.s[0]\n\t"
                    "mov v28.s[2], v30.s[0]\n\t"
                    "mov v28.s[3], v31.s[0]\n\t"
                    
                    // Store the 4 int32 results
                    "st1 {v16.4s},  [%[dst_1]], #16\n\t"
                    "st1 {v20.4s},  [%[dst_2]], #16\n\t"
                    "st1 {v24.4s},  [%[dst_3]], #16\n\t"
                    "st1 {v28.4s},  [%[dst_4]], #16\n\t"
                    
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
                return Status::Success;
            }
        }
    }
}
#endif

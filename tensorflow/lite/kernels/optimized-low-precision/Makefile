MKDIR=mkdir
TARGET_ISA=aarch64
RUY_LIB=/home/user9886/Project/Experiments/po2-multiplication-kernel/ruy/bazel-bin/ruy
RUY_INC=/home/user9886/Project/Experiments/po2-multiplication-kernel/ruy
RUY_CCFLAGS := 	-Wall -Wextra -Wc++14-compat -Wundef -lpthread
RUY_LIB_LINK := -lcontext -lkernel_arm -lpack_arm -lfrontend -lprepacked_cache -lcontext_get_ctx -lctx -lallocator -lcpuinfo -lthread_pool -lprepare_packed_matrices -ltrmul -lblock_map -lapply_multiplier -lblocking_counter -ldenormal -lsystem_aligned_alloc -ltune -lwait
CPU_LIB=/home/user9886/Project/Experiments/po2-multiplication-kernel/ruy/bazel-bin/external/cpuinfo
CPU_INC=/home/user9886/Project/Experiments/po2-multiplication-kernel/ruy/third_party/cpuinfo/include
CPU_LIB_LINK := -lcpuinfo_impl -lclog

KERNELS_OBJS := kernels/Int8-Int4.o kernels/Int4-Int8.o kernels/Int4-Int4.o kernels/Int8-Ternary.o kernels/Ternary-Int8.o kernels/Ternary-Ternary.o kernels/Int8-Binary.o kernels/Binary-Int8.o kernels/Binary-Binary.o kernels/Binary-Binary-XOR.o kernels/Int8-Quaternary.o kernels/Int3-Int3.o

LDFLAGS :=

DEBUG ?= 1
ifeq ($(DEBUG), 1)
    CCFLAGS = -static -pthread -lstdc++ -g -march=armv8.2-a+fp16 -Wno-pointer-arith -DIS_ARM -DTFLITE_BUILD -lm
else
    CCFLAGS = -static -pthread -lstdc++ -O2 -march=armv8.2-a+fp16 -Wno-pointer-arith -DIS_ARM -DTFLITE_BUILD -lm
endif

DISABLE_KERNELS_MEM_ACCESS ?= 0
ifeq ($(DISABLE_KERNELS_MEM_ACCESS), 1)
    KERNELS_MEM_ACCESS_FLAGS = -DDISABLE_KERNELS_MEM_ACCESS
else
    KERNELS_MEM_ACCESS_FLAGS = -UDISABLE_KERNELS_MEM_ACCESS
endif

CXX = /usr/bin/aarch64-linux-gnu-g++
CC = /usr/bin/aarch64-linux-gnu-gcc

all:												Build-Ruy \
													low_precision_fully_connected.o \
													ops-implementations/mul/LowPrecisionPacking.o \
													low_precision_fully_connected_test.o \
													test-16bit-2bit-packing \
													common/types.h \
													common/flags.h \
													common/half.hpp \
													common/asmutility.h \
													Makefile
	$(CC) low_precision_fully_connected.o ops-implementations/mul/LowPrecisionPacking.o low_precision_fully_connected_test.o $(KERNELS_OBJS) -L$(RUY_LIB) $(RUY_LIB_LINK) -L$(CPU_LIB) $(CPU_LIB_LINK) $(RUY_CCFLAGS) $(CCFLAGS) ${LDFLAGS} -o low_precision_fully_connected_test
Build-Ruy:					
	$(MAKE) -C /home/user9886/Project/Experiments/po2-multiplication-kernel/ruy DEBUG=$(DEBUG) DISABLE_KERNELS_MEM_ACCESS=$(DISABLE_KERNELS_MEM_ACCESS)

############################# Kernels Start #############################

kernels/Int8-Int4.o:								kernels/Int8-Int4.cc \
													low_precision_fully_connected.h \
													Makefile
	$(CC) kernels/Int8-Int4.cc $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Int8-Int4.o -c

kernels/Int4-Int8.o:								kernels/Int4-Int8.cc \
													low_precision_fully_connected.h \
													Makefile
	$(CC) kernels/Int4-Int8.cc $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Int4-Int8.o -c

kernels/Int4-Int4.o:								kernels/Int4-Int4.cc \
													low_precision_fully_connected.h \
													Makefile
	$(CC) kernels/Int4-Int4.cc $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Int4-Int4.o -c

kernels/Int8-Ternary.o:								kernels/Int8-Ternary.cc \
													low_precision_fully_connected.h \
													Makefile
	$(CC) kernels/Int8-Ternary.cc $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Int8-Ternary.o -c

kernels/Ternary-Int8.o:								kernels/Ternary-Int8.cc \
													low_precision_fully_connected.h \
													Makefile
	$(CC) kernels/Ternary-Int8.cc $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Ternary-Int8.o -c

kernels/Ternary-Ternary.o:							kernels/Ternary-Ternary.cc \
													low_precision_fully_connected.h \
													Makefile
	$(CC) kernels/Ternary-Ternary.cc $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Ternary-Ternary.o -c

kernels/Int8-Binary.o:								kernels/Int8-Binary.cc \
													low_precision_fully_connected.h \
													Makefile
	$(CC) kernels/Int8-Binary.cc $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Int8-Binary.o -c

kernels/Binary-Int8.o:								kernels/Binary-Int8.cc \
													low_precision_fully_connected.h \
													Makefile
	$(CC) kernels/Binary-Int8.cc $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Binary-Int8.o -c

kernels/Binary-Binary.o:							kernels/Binary-Binary.cc \
													low_precision_fully_connected.h \
													Makefile
	$(CC) kernels/Binary-Binary.cc $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Binary-Binary.o -c

kernels/Binary-Binary-XOR.o:						kernels/Binary-Binary-XOR.cc \
													low_precision_fully_connected.h \
													Makefile
	$(CC) kernels/Binary-Binary-XOR.cc $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Binary-Binary-XOR.o -c

kernels/Int8-Quaternary.o:							kernels/Int8-Quaternary.cc \
													low_precision_fully_connected.h \
													Makefile
	$(CC) kernels/Int8-Quaternary.cc $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Int8-Quaternary.o -c

kernels/Int3-Int3.o:								kernels/Int3-Int3.cc \
													low_precision_fully_connected.h \
													Makefile
	$(CC) kernels/Int3-Int3.cc $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Int3-Int3.o -c
#############################  Kernels End  #############################

low_precision_fully_connected.o:					low_precision_fully_connected.cc \
													$(KERNELS_OBJS) \
													low_precision_fully_connected.h \
													Makefile
	$(CC) low_precision_fully_connected.cc $(CCFLAGS) ${LDFLAGS} -o low_precision_fully_connected.o -c

ops-implementations/mul/LowPrecisionPacking.o:		ops-implementations/mul/LowPrecisionPacking.cc \
													ops-implementations/mul/LowPrecisionPacking.h \
													Makefile
	$(CC) ops-implementations/mul/LowPrecisionPacking.cc $(CCFLAGS) ${LDFLAGS} -o ops-implementations/mul/LowPrecisionPacking.o -c

low_precision_fully_connected_test.o:				low_precision_fully_connected_test.cc \
													low_precision_fully_connected_benchmark.h \
													Makefile
	$(CC) low_precision_fully_connected_test.cc  $(KERNELS_MEM_ACCESS_FLAGS) -I$(RUY_INC) $(CCFLAGS) ${LDFLAGS} -o low_precision_fully_connected_test.o -c

test-16bit-2bit-packing:							test-16bit-2bit-packing.o \
													Makefile
	$(CC) test-16bit-2bit-packing.o $(CCFLAGS) ${LDFLAGS} -o test-16bit-2bit-packing

test-16bit-2bit-packing.o:							test-16bit-2bit-packing.cc \
													Makefile
	$(CC) test-16bit-2bit-packing.cc $(CCFLAGS) ${LDFLAGS} -o test-16bit-2bit-packing.o -c

clean:
	rm -f \
		low_precision_fully_connected_test.o \
		low_precision_fully_connected.o \
		ops-implementations/mul/LowPrecisionPacking.o \
		low_precision_fully_connected_test \
		$(KERNELS_OBJS)
	$(MAKE) -C /home/user9886/Project/Experiments/po2-multiplication-kernel/ruy DEBUG=$(DEBUG) clean


MKDIR=mkdir
TARGET_ISA=aarch64
RUY_LIB=/home/user9886/Project/Experiments/po2-multiplication-kernel/ruy/bazel-bin/ruy
RUY_INC=/home/user9886/Project/Experiments/po2-multiplication-kernel/ruy
RUY_CCFLAGS := 	-Wall \
				-Wextra \
				-Wc++14-compat \
				-Wundef \
				-lpthread
RUY_LIB_LINK := -lcontext \
				-lkernel_arm \
				-lpack_arm \
				-lfrontend \
				-lprepacked_cache \
				-lcontext_get_ctx \
				-lctx \
				-lallocator \
				-lcpuinfo \
				-lthread_pool \
				-lprepare_packed_matrices \
				-ltrmul \
				-lblock_map \
				-lapply_multiplier \
				-lblocking_counter \
				-ldenormal \
				-lsystem_aligned_alloc \
				-ltune \
				-lwait

CPU_LIB=/home/user9886/Project/Experiments/po2-multiplication-kernel/ruy/bazel-bin/external/cpuinfo
CPU_INC=/home/user9886/Project/Experiments/po2-multiplication-kernel/ruy/third_party/cpuinfo/include
CPU_LIB_LINK := -lcpuinfo_impl -lclog

LDFLAGS :=
DEBUG ?= 1
ifeq ($(DEBUG), 1)
    CCFLAGS = -static -pthread -lstdc++ -g -march=armv8.2-a+fp16 -DIS_ARM -DTFLITE_BUILD
else
    CCFLAGS = -static -pthread -lstdc++ -O2 -march=armv8.2-a+fp16 -DIS_ARM -DTFLITE_BUILD
endif
CXX = /usr/bin/aarch64-linux-gnu-g++
CC = /usr/bin/aarch64-linux-gnu-gcc

all:												Build-Ruy \
													low_precision_fully_connected.o \
													ops-implementations/mul/LowPrecisionPacking.o \
													low_precision_fully_connected_test.o
	$(CC) \
		low_precision_fully_connected.o \
		ops-implementations/mul/LowPrecisionPacking.o \
		low_precision_fully_connected_test.o \
		-L$(RUY_LIB) $(RUY_LIB_LINK) \
		-L$(CPU_LIB) $(CPU_LIB_LINK) \
		$(RUY_CCFLAGS) \
		$(CCFLAGS) ${LDFLAGS} \
		-o low_precision_fully_connected_test
Build-Ruy:					
	$(MAKE) -C /home/user9886/Project/Experiments/po2-multiplication-kernel/ruy DEBUG=$(DEBUG)

low_precision_fully_connected.o:					low_precision_fully_connected.cc \
													low_precision_fully_connected.h
	$(CC) low_precision_fully_connected.cc $(CCFLAGS) ${LDFLAGS} -o low_precision_fully_connected.o -c

ops-implementations/mul/LowPrecisionPacking.o:		ops-implementations/mul/LowPrecisionPacking.cc \
													ops-implementations/mul/LowPrecisionPacking.h
	$(CC) ops-implementations/mul/LowPrecisionPacking.cc $(CCFLAGS) ${LDFLAGS} -o ops-implementations/mul/LowPrecisionPacking.o -c

low_precision_fully_connected_test.o:				low_precision_fully_connected_test.cc
	$(CC) low_precision_fully_connected_test.cc -I$(RUY_INC) $(CCFLAGS) ${LDFLAGS} -o low_precision_fully_connected_test.o -c

clean:
	rm -f \
		low_precision_fully_connected_test.o \
		low_precision_fully_connected.o \
		ops-implementations/mul/LowPrecisionPacking.o \
		low_precision_fully_connected_test


BAZEL=bazel

DEBUG ?= 1
ifeq ($(DEBUG), 1)
    BUILD_TYPE = dbg
else
    BUILD_TYPE = opt
endif

DISABLE_KERNELS_MEM_ACCESS ?= 1
ifeq ($(DISABLE_KERNELS_MEM_ACCESS), 1)
    KERNELS_MEM_ACCESS_FLAGS = -DDISABLE_KERNELS_MEM_ACCESS
else
    KERNELS_MEM_ACCESS_FLAGS = -UDISABLE_KERNELS_MEM_ACCESS
endif

all:
	$(shell $(BAZEL) build --jobs `nproc` --config=aarch64 --copt=$(KERNELS_MEM_ACCESS_FLAGS) --cxxopt=$(KERNELS_MEM_ACCESS_FLAGS) -c $(BUILD_TYPE) //example:multiplication-example)
clean:
	$(shell $(BAZEL) clean)
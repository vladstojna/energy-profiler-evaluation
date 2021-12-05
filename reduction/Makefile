prefix  := reduction
target  := $(prefix).out
src     := $(prefix).cu
obj     := $(patsubst %.cu, %.o, $(src))
deps    := $(patsubst %.cu, %.d, $(src))

incls := ../timeprinter/include ../include
libs  := ../timeprinter/lib

nvcc := nvcc
nvcc_ver := $(shell $(nvcc) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
cuda_major := $(firstword $(subst ., ,$(nvcc_ver)))
cuda_too_old := $(shell test $(cuda_major) -ge 9; echo $$?)

ifeq (1,$(cuda_too_old))
$(error CUDA version 9 or later required)
endif

ifeq (11,$(cuda_major))
std := c++17
timeprinter := libtimeprinter.a
else
std := c++14
timeprinter := libtimeprinter.$(std).a
endif

CUDA_ROOT ?= /opt/cuda
incls += $(CUDA_ROOT)/include
libs += $(CUDA_ROOT)/lib$(shell getconf LONG_BIT)

cpp := THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP
cpp += THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA

cflags  := -g
cflags  += --compiler-options -Wall,-Wextra,-Wunknown-pragmas,-fPIE,-fopenmp
cflags  += -std=$(std)
cflags  += $(addprefix -I, $(incls))
cflags  += $(addprefix -D, $(cpp))

ld      := g++
ldflags += $(addprefix -L, $(libs))
ldflags += -l:$(timeprinter) -lpthread -lcurand -lcudart -lgomp

debug ?=
ifdef debug
cflags += -O0
else
cflags += -O3 -DNDEBUG
endif

.PHONY: default remake clean

default: $(target)

$(target): $(obj)
	$(ld) $^ $(ldflags) -o $@

%.o: %.cu %.d
	$(nvcc) -MT $@ -MMD -MP -MF $*.d $(cflags) -c -o $@ $<

$(deps):

include $(wildcard $(deps))

remake: clean default

clean:
	rm -rf $(target) $(deps) $(obj)

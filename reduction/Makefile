prefix  := reduction
target  := $(prefix).out
src     := $(prefix).cu
obj     := $(patsubst %.cu, %.o, $(src))
deps    := $(patsubst %.cu, %.d, $(src))

incls := ../timeprinter/include ../include
libs  := ../timeprinter/lib

CUDA_ROOT ?= /opt/cuda
incls += $(CUDA_ROOT)/include
libs += $(CUDA_ROOT)/lib$(shell getconf LONG_BIT)

cpp := THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP
cpp += THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA

cc      := nvcc
cflags  := -g
cflags  += --compiler-options -Wall,-Wextra,-Wunknown-pragmas,-fPIE,-fopenmp
cflags  += -std=c++17
cflags  += $(addprefix -I, $(incls))
cflags  += $(addprefix -D, $(cpp))

ld      := g++
ldflags += $(addprefix -L, $(libs))
ldflags += -l:libtimeprinter.a -lpthread -lcurand -lcudart -lgomp

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
	$(cc) -MT $@ -MMD -MP -MF $*.d $(cflags) -c -o $@ $<

$(deps):

include $(wildcard $(deps))

remake: clean default

clean:
	rm -rf $(target) $(deps) $(obj)

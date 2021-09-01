target  := lapacke.out
src     := $(wildcard *.cpp)
obj     := $(patsubst %.cpp, %.o, $(src))
deps    := $(patsubst %.cpp, %.d, $(src))

mkl   := intel-mkl
lpk   := lapacke
impl  ?= $(lpk)
ifeq (,$(filter $(impl), $(lpk) $(mkl)))
$(error unsupported implementation $(impl))
endif

incls := ../timeprinter/include ../include
libs  := ../timeprinter/lib
cpp   :=

ifeq ($(impl), $(mkl))
incls += /opt/intel/mkl/include
libs  += $(MKLROOT)/lib/intel64
cpp   += MKL_ILP64 V_USE_MKL
else
cpp   += V_USE_BASELINE
endif

cc      := g++
cflags  := -Wall -Wextra -Wunknown-pragmas -pedantic -fPIE -g
cflags  += -std=c++17
cflags  += $(addprefix -I, $(incls))
cflags  += $(addprefix -D, $(cpp))

ifeq ($(impl), $(mkl))
cflags += -m64
endif

ldflags := $(addprefix -L, $(libs))
ldflags += -l:libtimeprinter.a -lpthread

ifeq ($(impl), $(mkl))
ldflags += -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lm -ldl
else
ldflags += -llapacke
endif

debug ?=
ifdef debug
cflags += -Og
else
cflags += -O3 -DNDEBUG
endif

.PHONY: default remake clean

default: $(target)

$(target): $(obj)
	$(cc) $^ $(ldflags) -o $@

%.o: %.cpp %.d
	$(cc) -MT $@ -MMD -MP -MF $*.d $(cflags) -c -o $@ $<

$(deps):

include $(wildcard $(deps))

remake: clean default

clean:
	rm -rf $(target) $(deps) $(obj)

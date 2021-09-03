target  := cblas.out
src     := $(wildcard *.cpp)
obj     := $(patsubst %.cpp, %.o, $(src))
deps    := $(patsubst %.cpp, %.d, $(src))

mkl   := intel-mkl
oblas := openblas
impl  ?= $(oblas)
ifeq (,$(filter $(impl), $(oblas) $(mkl)))
$(error unsupported implementation $(impl))
endif

incls := ../timeprinter/include ../include
libs  := ../timeprinter/lib
cpp   :=

NO_CBLAS_SYMS ?=
USE_RPATH ?=
rpath :=

ifeq ($(impl), $(mkl))
incls += $(MKLROOT)/include
libs  += $(MKLROOT)/lib/intel64
cpp   += MKL_ILP64 V_USE_MKL

ifdef USE_RPATH
rpath += -Wl,-rpath,$(MKLROOT)/lib/intel64
endif # USE_RPATH

else

ifdef OPENBLAS_ROOT
incls += $(OPENBLAS_ROOT)/include
libs  += $(OPENBLAS_ROOT)/lib
endif # OPENBLAS_ROOT
cpp   += V_USE_OPENBLAS

ifdef USE_RPATH
rpath += -Wl,-rpath,$(OPENBLAS_ROOT)/lib
endif # USE_RPATH

endif # ($(impl), $(mkl))

cc      := g++
cflags  := -Wall -Wextra -Wunknown-pragmas -pedantic -fPIE -g
cflags  += -std=c++17
cflags  += $(addprefix -I, $(incls))
cflags  += $(addprefix -D, $(cpp))

ifeq ($(impl), $(mkl))
cflags += -m64
endif

ldflags += $(addprefix -L, $(libs))
ldflags += -l:libtimeprinter.a -lpthread
ldflags += $(rpath)

ifeq ($(impl), $(mkl))
ldflags += -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lm -ldl
else
ldflags += -lopenblas

ifdef NO_CBLAS_SYMS
ldflags += -lcblas
endif

endif # ($(impl), $(mkl))

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

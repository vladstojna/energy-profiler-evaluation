prefix  := cusparse
target  := $(prefix).out
src     := $(prefix).cpp
obj     := $(patsubst %.cpp, %.o, $(src))
deps    := $(patsubst %.cpp, %.d, $(src))

incls := ../timeprinter/include ../include
libs  := ../timeprinter/lib

CUDA_ROOT ?= /opt/cuda
incls += $(CUDA_ROOT)/include
libs += $(CUDA_ROOT)/lib$(shell getconf LONG_BIT)

cc      := g++
cflags  := -Wall -Wextra -Wunknown-pragmas -pedantic -fPIE -g
cflags  += -std=c++17
cflags  += $(addprefix -I, $(incls))
cflags  += $(addprefix -D, $(cpp))

ldflags += $(addprefix -L, $(libs))
ldflags += -l:libtimeprinter.a -lpthread -lcusparse -lcudart

debug ?=
ifdef debug
cflags += -Og
else
cflags += -O3 -DNDEBUG -fno-optimize-sibling-calls
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

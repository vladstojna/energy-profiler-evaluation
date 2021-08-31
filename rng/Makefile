target  := rng.out
src     := $(wildcard *.cpp)
obj     := $(patsubst %.cpp, %.o, $(src))
deps    := $(patsubst %.cpp, %.d, $(src))

incls := ../timeprinter/include
libs  := ../timeprinter/lib

cc      := g++
cflags  := -Wall -Wextra -Wunknown-pragmas -pedantic -fPIE -fopenmp -g
cflags  += -std=c++17
cflags  += $(addprefix -I, $(incls))
cflags  += $(addprefix -D, $(cpp))

ldflags := $(addprefix -L, $(libs))
ldflags += -l:libtimeprinter.a -fopenmp

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

target  := cblas.out
src     := $(wildcard *.cpp)
obj     := $(patsubst %.cpp, %.o, $(src))
deps    := $(patsubst %.cpp, %.d, $(src))

incls   := ../timeprinter/include
libs    := ../timeprinter/lib

cc      := g++
cflags  := -Wall -Wextra -Wunknown-pragmas -pedantic -fPIE -pthread
cflags  += -std=c++17
cflags  += $(addprefix -I, $(incls))

ldflags := $(addprefix -L, $(libs))
ldflags += -lopenblas -lcblas -lpthread -l:libtimeprinter.a

debug ?=
ifdef debug
cflags += -O0 -g
else
cflags += -O3
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

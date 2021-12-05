src_dir  := src
incl_dir := include
obj_dir  := obj
dep_dir  := $(obj_dir)/.deps
tgt_dir  := lib

std ?= c++17
obj_suffix := $(if $(filter c++17,$(strip $(std))),,.$(strip std))

supported_stds := c++14 c++17
ifeq (,$(findstring $(strip $(std)),$(supported_stds)))
$(error Supported C++ standards: $(supported_stds))
endif

target := $(tgt_dir)/libtimeprinter$(obj_suffix)
src    := $(wildcard src/*.cpp)
obj    := $(patsubst $(src_dir)/%.cpp, $(obj_dir)/%$(obj_suffix).o, $(src))
deps   := $(patsubst $(src_dir)/%.cpp, $(dep_dir)/%$(obj_suffix).d, $(src))

system_clock ?=
ifdef system_clock
cpp := TP_USE_SYSTEM_CLOCK
endif

cc      := g++
cflags  := -Wall -Wextra -Wunknown-pragmas -pedantic -fPIC -pthread
cflags  += -std=$(strip $(std))
cflags  += $(addprefix -I, $(incl_dir))
cflags  += $(addprefix -D, $(cpp))

ldflags += -pthread -shared

ar := ar
arflags := -crs

debug ?=
ifdef debug
cflags += -O0 -g
else
cflags += -O3
endif

.PHONY: default remake clean dynamic static

default: dynamic

$(tgt_dir):
	@mkdir -p $@
$(obj_dir):
	@mkdir -p $@
$(dep_dir):
	@mkdir -p $@

dynamic: $(target).so

static: $(target).a

$(target).so: $(obj) | $(tgt_dir)
	$(cc) $^ $(ldflags) -o $@

$(target).a: $(obj) | $(tgt_dir)
	$(ar) $(arflags) $@ $^

$(obj_dir)/%$(obj_suffix).o: $(src_dir)/%.cpp $(dep_dir)/%$(obj_suffix).d | $(obj_dir) $(dep_dir)
	$(cc) -MT $@ -MMD -MP -MF $(dep_dir)/$*$(obj_suffix).d $(cflags) -c -o $@ $<

$(deps):

include $(wildcard $(deps))

remake: clean default

clean:
	rm -rf $(tgt_dir) $(obj_dir)

src_dir  := src
incl_dir := include
obj_dir  := obj
dep_dir  := $(obj_dir)/.deps
tgt_dir  := lib

target := $(tgt_dir)/libtimeprinter
src    := $(wildcard src/*.cpp)
obj    := $(patsubst $(src_dir)/%.cpp, $(obj_dir)/%.o, $(src))
deps   := $(patsubst $(src_dir)/%.cpp, $(dep_dir)/%.d, $(src))

system_clock ?=
ifdef system_clock
cpp := TP_USE_SYSTEM_CLOCK
endif

cc      := g++
cflags  := -Wall -Wextra -Wunknown-pragmas -pedantic -fPIC -pthread
cflags  += -std=c++17
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

$(obj_dir)/%.o: $(src_dir)/%.cpp $(dep_dir)/%.d | $(obj_dir) $(dep_dir)
	$(cc) -MT $@ -MMD -MP -MF $(dep_dir)/$*.d $(cflags) -c -o $@ $<

$(deps):

include $(wildcard $(deps))

remake: clean default

clean:
	rm -rf $(tgt_dir) $(obj_dir)

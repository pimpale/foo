TARGET_EXEC ?= "hello-world"

BUILD_DIR ?= ./obj
SRC_DIRS ?= src

SRCS := $(shell find $(SRC_DIRS) -type f -name *.ll)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

INC_DIRS := include
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

LDFLAGS := -lm -no-pie

CC := clang
CFLAGS ?= $(INC_FLAGS) -std=c2x -MMD -MP -O0 -g3 -Wall -Weverything -pedantic -Wno-switch-enum -Wno-unsafe-buffer-usage -Wno-declaration-after-statement -Wno-pre-c23-compat

LLC := llc
LLCFLAGS ?= --filetype=obj

#CC := clang
#CFLAGS ?= $(INC_FLAGS) -std=c11 --analyze -MMD -MP -O0 -g3 -Wall -Weverything -pedantic -Wno-padded -Wno-switch-enum 

#CC := gcc
#CFLAGS ?= $(INC_FLAGS) -fanalyzer -std=c11 -MMD -MP -O0 -g3 -Wall -pedantic -Wno-padded -Wno-switch-enum

#CC := afl-gcc
#CFLAGS ?= $(INC_FLAGS) -std=c11 -MMD -MP -O0 -g3 -Wall -pedantic -Wno-padded -Wno-switch-enum

$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

# c source
$(BUILD_DIR)/%.c.o: %.c
	$(MKDIR_P) $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# llvm ir source
$(BUILD_DIR)/%.ll.o: %.ll
	$(MKDIR_P) $(dir $@)
	$(LLC) $(LLCFLAGS) $< -o $@


.PHONY: clean
clean:
	$(RM) -r $(BUILD_DIR)


-include $(DEPS)

MKDIR_P ?= mkdir -p


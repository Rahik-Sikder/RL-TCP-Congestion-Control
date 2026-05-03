CC      = gcc
# gnu11 exposes POSIX symbols (clock_gettime, CLOCK_MONOTONIC) missing in strict c11
STD     = -std=gnu11
WARN    = -Wall -Wextra
OPT     = -O2
CFLAGS  = $(STD) $(WARN) $(OPT)
DBGFLAGS = $(STD) $(WARN) -O0 -g -DDEBUG

# Prefer local tar-release, then Homebrew (macOS), then system /usr
ONNX_LOCAL := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))../onnxruntime)
ONNX_PREFIX ?= $(shell \
  if [ -d "$(ONNX_LOCAL)/include" ]; then echo "$(ONNX_LOCAL)"; \
  else brew --prefix onnxruntime 2>/dev/null || echo /usr; fi)

INCS    = -I. -Itransport -Icc -I$(ONNX_PREFIX)/include
LIBS    = -lm
# Linker flags for sender (requires ONNX Runtime)
# -rpath embeds the library search path in the binary so LD_LIBRARY_PATH isn't needed at runtime
SENDER_LIBS = $(LIBS) -L$(ONNX_PREFIX)/lib -lonnxruntime -Wl,-rpath,$(ONNX_PREFIX)/lib

TRANSPORT_SRCS = transport/transport.c transport/queue.c
CC_SRCS        = cc/new_reno.c cc/cubic.c cc/ppo.c
SENDER_SRCS    = sender.c $(TRANSPORT_SRCS) $(CC_SRCS)
RECEIVER_SRCS  = receiver.c $(TRANSPORT_SRCS)

KATHARA_DIR    := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))../kathara)
KATHARA_SHARED := $(KATHARA_DIR)/shared
SCENARIOS      := bottleneck high_loss simple_p2p variable_latency
SCENARIO_DIRS  := $(addprefix $(KATHARA_DIR)/scenarios/,$(SCENARIOS))
ONNX_LIB_DIR   := $(ONNX_PREFIX)/lib

.PHONY: all debug clean install

all: sender receiver install

sender: $(SENDER_SRCS)
	$(CC) $(CFLAGS) $(INCS) -o $@ $^ $(SENDER_LIBS)

receiver: $(RECEIVER_SRCS)
	$(CC) $(CFLAGS) $(INCS) -o $@ $^ $(LIBS)

debug: $(SENDER_SRCS) $(RECEIVER_SRCS)
	$(CC) $(DBGFLAGS) $(INCS) -o sender $(SENDER_SRCS) $(SENDER_LIBS)
	$(CC) $(DBGFLAGS) $(INCS) -o receiver $(RECEIVER_SRCS) $(LIBS)
	$(MAKE) install

install: sender receiver
	@for d in $(SCENARIO_DIRS); do \
	    mkdir -p "$$d/shared/lib"; \
	    cp -f sender receiver $(KATHARA_SHARED)/run_sender.sh $(KATHARA_SHARED)/run_receiver.sh "$$d/shared/"; \
	    chmod +x "$$d/shared/sender" "$$d/shared/receiver" "$$d/shared/run_sender.sh" "$$d/shared/run_receiver.sh"; \
	    cp -Pf $(ONNX_LIB_DIR)/libonnxruntime.so* "$$d/shared/lib/" 2>/dev/null || true; \
	done

clean:
	rm -f sender receiver
	@for d in $(SCENARIO_DIRS); do \
	    rm -f "$$d/shared/sender" "$$d/shared/receiver" \
	          "$$d/shared/run_sender.sh" "$$d/shared/run_receiver.sh"; \
	    rm -rf "$$d/shared/lib"; \
	done

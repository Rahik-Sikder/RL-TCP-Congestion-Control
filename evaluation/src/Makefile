CC      = gcc
STD     = -std=c11
WARN    = -Wall -Wextra
OPT     = -O2
CFLAGS  = $(STD) $(WARN) $(OPT)
DBGFLAGS = $(STD) $(WARN) -O0 -g -DDEBUG

# Auto-detect ONNX Runtime prefix (Homebrew macOS or system Linux)
ONNX_PREFIX ?= $(shell brew --prefix onnxruntime 2>/dev/null || echo /usr)

INCS    = -I. -Itransport -Icc -I$(ONNX_PREFIX)/include
LIBS    = -lm
# Linker flags for sender (requires ONNX Runtime)
SENDER_LIBS = $(LIBS) -L$(ONNX_PREFIX)/lib -lonnxruntime

TRANSPORT_SRCS = transport/transport.c transport/queue.c
CC_SRCS        = cc/new_reno.c cc/cubic.c cc/ppo.c
SENDER_SRCS    = sender.c $(TRANSPORT_SRCS) $(CC_SRCS)
RECEIVER_SRCS  = receiver.c $(TRANSPORT_SRCS)

.PHONY: all debug clean

all: sender receiver

sender: $(SENDER_SRCS)
	$(CC) $(CFLAGS) $(INCS) -o $@ $^ $(SENDER_LIBS)

receiver: $(RECEIVER_SRCS)
	$(CC) $(CFLAGS) $(INCS) -o $@ $^ $(LIBS)

debug: $(SENDER_SRCS) $(RECEIVER_SRCS)
	$(CC) $(DBGFLAGS) $(INCS) -o sender $(SENDER_SRCS) $(SENDER_LIBS)
	$(CC) $(DBGFLAGS) $(INCS) -o receiver $(RECEIVER_SRCS) $(LIBS)

clean:
	rm -f sender receiver

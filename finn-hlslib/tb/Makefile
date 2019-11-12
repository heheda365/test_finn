

CINCLUDES	+= -I../
CINCLUDES 	+= -I${XILINX_VIVADO}/include/
CINCLUDES 	+= -std=c++0x -Wall -Wno-unknown-pragmas -Wall -Wno-unknown-pragmas -Wno-unused-variable -g
CXX			= g++
LDFLAGS		= -lpthread

# all: add_tb

test: conv_stream_tb.cpp
	$(CXX) $(CFLAGS) $(CINCLUDES) conv_stream_tb.cpp -o test $(LDFLAGS)
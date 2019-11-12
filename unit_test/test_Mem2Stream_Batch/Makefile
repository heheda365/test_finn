

CINCLUDES	+= -I../../finn-hlslib/
CINCLUDES 	+= -I${XILINX_VIVADO}/include/
CINCLUDES 	+= -std=c++0x -Wall -Wno-unknown-pragmas -Wall -Wno-unknown-pragmas -Wno-unused-variable -g
CXX			= g++
LDFLAGS		= -lpthread


test: test_Mem2Stream_Batch.cpp
	$(CXX) $(CFLAGS) $(CINCLUDES) test_Mem2Stream_Batch.cpp -o test $(LDFLAGS)
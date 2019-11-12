

CINCLUDES	+= -I../../finn-hlslib/
CINCLUDES 	+= -I${XILINX_VIVADO}/include/
CINCLUDES 	+= -std=c++0x -Wall -Wno-unknown-pragmas -Wall -Wno-unknown-pragmas -Wno-unused-variable -g
CXX			= g++
LDFLAGS		= -lpthread


test: test_activation.cpp
	$(CXX) $(CFLAGS) $(CINCLUDES) test_activation.cpp -o test $(LDFLAGS)
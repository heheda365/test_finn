
#include "bnn-library.h"
#include <iostream>


int main(int argc, char const *argv[])
{
    const unsigned int inBits = 32 * 32 * 3 * 8;
    ap_uint<64> in[32 * 32 * 3 / 8];

    for(int i=0; i < 32 * 32 * 3 / 8; i ++) {
        in[i] = i;
        // std::cout << in[i] << "  ";
    }
    std::cout << std::endl;
    // std::cout << sizeof(in[0]);

    stream<ap_uint<64>> inter0("DoCompute.inter0");
    Mem2Stream_Batch<64, inBits / 8>(in, inter0, 2);

    for(int i=0; i < 64; i ++) {
        std::cout << inter0.read() << "  ";
    }

    return 0;
}

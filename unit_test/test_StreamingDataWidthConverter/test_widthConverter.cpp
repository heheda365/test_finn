
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
    Mem2Stream_Batch<64, inBits / 8>(in, inter0, 1);

    // for(int i=0; i < 32 * 32 * 3 / 8; i ++) {
    //     std::cout << inter0.read() << "  ";
    // }
    // std::cout << std::endl;

    stream<ap_uint<192>> inter0_1("DoCompute.inter0_1");
    stream<ap_uint<24>> inter0_2("DoCompute.inter0_2");

    

    StreamingDataWidthConverter_Batch<64, 192, (32 * 32 * 3 * 8) / 64>(inter0, inter0_1, 1);
    StreamingDataWidthConverter_Batch<192, 24, (32 * 32 * 3 * 8) / 192>(inter0_1, inter0_2, 1);

    for(int i=0; i < 50; i ++) {
        // in[i] = i;
        // std::cout << in[i] << "  ";
        std::cout << inter0_2.read() << "  ";
    }

    return 0;
}

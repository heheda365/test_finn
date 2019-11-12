
#include "bnn-library.h"
#include "activations.hpp"
#include "weights.hpp"
// #include "activations.hpp"
#include "interpret.hpp"
#include "mvau.hpp"
#include <iostream>




template<
  unsigned MatrixW, unsigned MatrixH, unsigned SIMD, unsigned PE,
  typename TSrcI = Identity, typename TDstI = Identity, typename TWeightI = Identity,
  typename TI, typename TO, typename TW, typename TA, typename R
>
void Matrix_Vector_Activate_Batch_test(hls::stream<TI> &in,
				  hls::stream<TO> &out,
				  TW  const &weights,
				  TA  const &activation,
				  int const  reps,
				  R const &r) {

  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  unsigned const  NF = MatrixH / PE;

  // how many synapse groups each row is split into
  // alternatively: number of horizontal matrix chunks
  unsigned const  SF = MatrixW / SIMD;

  // input vector buffers
  TI  inputBuf[SF];
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1

  decltype(activation.init(0,0))  accu[PE];
#pragma HLS ARRAY_PARTITION variable=accu complete dim=0

  unsigned  nf   = 0;
  unsigned  sf   = 0;
  unsigned  tile = 0; // invariant: tile = nf*SF + sf

  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelinening the way we want
  unsigned const TOTAL_FOLD = NF * SF;
  for(unsigned  i = 0; i < reps * TOTAL_FOLD; i++) {
#pragma HLS PIPELINE II=1
    TI  inElem;
    if(nf == 0) {
      // read input from stream
      inElem = in.read();
      // store in appropriate buffer for reuse
      inputBuf[sf] = inElem;
    }
    else {
      // reuse buffered input
      inElem = inputBuf[sf];
    }

    // Threshold Initialisation
    if(sf == 0) {
      for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
	    accu[pe] = activation.init(nf, pe);
      }
    }

    // compute matrix-vector product for each processing element
    auto const &w = weights.weights(tile);
    for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
      auto const  wgt = TWeightI()(w[pe]);
      auto const  act = TSrcI()(inElem);
      accu[pe] = mac<SIMD>(accu[pe], wgt, act, r); 
    //   std::cout << "tile" << tile << ": ";
    //   for(int i=0; i < SIMD; i ++) {
    //       std::cout << "pe" << pe << ":" << wgt[i] << " ";
    //   }
    }
    // std::cout << std::endl;

    // keep track of which folded synapse/neuron we are processing
    ++tile;
    if(++sf == SF) {
      // produce output and clear accumulators
      auto  outElem = TDstI().template operator()<TO>();
    //   std::cout << "out no act \n";
      for (unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
	    outElem[pe] = activation.activate(nf, pe, accu[pe]);
	    std::cout << accu[pe] << " ";
      }
      std::cout << "\n";

      out.write(outElem);

      // next folded neuron or image
      sf = 0;
      if(++nf == NF) {
	    nf   = 0;
	    tile = 0;
      }
    }
  }
}

#define K 4
#define K 4
#define IN_CH 2
#define MatrixW 32
#define MatrixH 8
#define SIMD 2
#define PE 2
#define OFMDim 8
int main(int argc, char const *argv[])
{
    ap_int<2> w_mat[32][8];
    for(int i=0; i < 32 / 4; i ++) {
        for(int j=0; j < 4; j ++) {
            w_mat[4 * i + 0][j] = -2; 
            w_mat[4 * i + 1][j] = -1;
            w_mat[4 * i + 2][j] = 0;
            w_mat[4 * i + 3][j] = 1;
        }
        for(int j=4; j < 8; j ++) {
            w_mat[4 * i + 0][j] = 1; 
            w_mat[4 * i + 1][j] = 0;
            w_mat[4 * i + 2][j] = 1;
            w_mat[4 * i + 3][j] = -1;
        }
    }
    std::cout << "w_mat = \n";
    for(int i=0; i < 32; i ++) {
        for(int j=0; j < 8; j ++) {
            std::cout << w_mat[i][j] << "  ";
        }
        std::cout << "\n";
    }
    // ap_uint<8> a = 1;
    // a(3, 2) = -1;
    // std::cout << "a = " << a << "\n";
    
    static FixedPointWeights<SIMD, ap_int<2>, PE, 64>   weights1;
    int tile = 0;
    for(int i=0; i < MatrixH / PE; i ++) {
        for(int j=0; j < MatrixW / SIMD; j ++) {

            for(int pe=0; pe < PE; pe ++) {
                ap_uint<SIMD*2> item = 0;
                for(int simd=0; simd < SIMD; simd ++) {
                    item(simd * 2 + 1, simd * 2) = w_mat[j*SIMD + simd][i*PE + pe]; 
                }
                weights1.m_weights[pe][tile] = item;
            }
            tile ++;
        }
    }
    std::cout << "weights1.m_weights = \n";
    for(int i=0; i < PE; i ++) {
        for(int j=0; j < 64; j ++) {
            std::cout << weights1.m_weights[i][j] << " ";
        }
        std::cout << "\n";
    }
    auto const &w = weights1.weights(32);
    auto const  wgt = Identity()(w[0]);
    for(int i=0; i < 2; i ++) {
        std::cout << wgt[i] << " ";
    }
    std::cout << "\n";

    


    static ThresholdsActivation<4, 2, 2, ap_int<16>, ap_int<2>, -1>  threshs1;

    for(int i=0; i < PE; i ++) {
        for(int j=0; j < 8 / PE; j ++) {
            for(int k=0; k < 2; k ++) {
                threshs1.m_thresholds[i][j][k] = 3;
            }
        }
    }
    ap_int<2> in_data[OFMDim * OFMDim][32];
    hls::stream<ap_uint<SIMD*2>> convInp("in");

    for(int i=0; i <  OFMDim * OFMDim; i ++) {
        for(int j=0; j < 32 / 2; j ++) {
            in_data[i][j * 2] = -1;
            in_data[i][j * 2 + 1] = 1;

            ap_uint<SIMD * 2> item = 0;
            item(1, 0) = in_data[i][j * 2];
            item(3, 2) = in_data[i][j * 2 + 1];
            convInp.write(item);
        }
    }
    std::cout << "res :\n";
    for(int i=0; i < OFMDim * OFMDim; i ++) {
        for(int col=0; col < 8; col ++) {
            int acc = 0;
            for(int j=0; j < 32; j ++) {
                acc += in_data[i][j] * w_mat[j][col];
            }
            std::cout << acc << "  ";
        }
        std::cout << "\n";
    }

    
    hls::stream<ap_uint<PE*2>> mvOut("out");

    std::cout << "out \n";
    Matrix_Vector_Activate_Batch_test<MatrixW, MatrixH, SIMD, PE, Slice<ap_int<2>>, Slice<ap_int<2>>, Identity>
    (static_cast<hls::stream<ap_uint<SIMD*2>>&>(convInp),
     static_cast<hls::stream<ap_uint<PE*2>>&>  (mvOut),
     weights1, threshs1, 1* OFMDim * OFMDim, ap_resource_lut());

    

    return 0;
}

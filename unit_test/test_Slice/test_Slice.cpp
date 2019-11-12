
#include "bnn-library.h"
#include <iostream>

template< typename TSrcI = Identity > 
void fun() {

    ap_uint<8>  inElem = 0x12;
    auto const  act = TSrcI()(inElem);
    
    std::cout << act[0] << "\n";
    std::cout << act[1] << "\n";
    std::cout << act[2] << "\n";
    std::cout << act[3] << "\n";
    // std::cout << act;
}



class A {
public:
    int operator() ( int val ){
        return val > 0 ? val : -val;
    }
    int static fun() {
        return 10;
    }
};


int main(int argc, char const *argv[])
{
    // fun< Slice<ap_uint<2>>> ();
    // ap_uint<8>  inElem = 0x12;
    // Slice<ap_uint<2>> a;
    // auto const  act = a(inElem);
    // std::cout << act[0] << "\n";
    // std::cout << act[1] << "\n";
    // std::cout << act[2] << "\n";
    // std::cout << act[3] << "\n";
    // Slice<ap_int<2>>()();
    int a = A()(-5);
    int b = A::fun();
    std::cout << a << "\n" << b << "\n";
    

    return 0;
}


#include "bnn-library.h"
#include "activations.hpp"
#include <iostream>



int main(int argc, char const *argv[])
{

    static ThresholdsActivation<2, 2, 2, ap_int<2>, ap_int<2>, -1>  threshs1;
    std::cout << typeid(threshs1.init(0,0)).name();
    

    return 0;
}

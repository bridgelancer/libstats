#include <iostream>
#include <armadillo>

#include "ADF/adf.h"

using namespace std;
using namespace arma;

int main()
{
    OPTIONS options;
    adf testing = adf(OPTIONS::DF);
    
    arma_rng::set_seed_random();

    vec y(10); y.randu();

    testing.setObservation(y);
     
    testing.evaluatePhi(OPTIONS::ADF,5);    

  


    return 0;   
}

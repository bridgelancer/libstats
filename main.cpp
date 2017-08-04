#include <iostream>
#include <armadillo>

#include "ADF/adf.h"

using namespace std;
using namespace arma;

int main()
{
    OPTIONS options;
    adf testing = adf(OPTIONS::ADF);
    
    arma_rng::set_seed_random();

    vec y(100); y.randu();

    testing.setObservation(y);
     
    testing.evaluateSE(5);    
    
  


    return 0;   
}

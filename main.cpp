#include <iostream>
#include <armadillo>

#include "ADF/adf.h"

using namespace std;
using namespace arma;

int main()
{
    arma_rng:: set_seed_random();
    OPTIONS options;
    adf testing = adf(OPTIONS::ADF);
   
    testing.loadCSV("gamble.mat");    
//    arma::vec y(10000);
//    y.randu();
    
//    testing.setObservation(y);
       
    testing.evaluateSE(0);     

    arma::vec beta = testing.regression.getBeta();
    beta.print("beta:");
     
    testing.getStatistics(); 

    return 0;   
}

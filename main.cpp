#include <iostream>
#include <armadillo>

#include "ADF/adf.h"

using namespace std;
using namespace arma;

int main()
{
    OPTIONS options;
    adf testing = adf(OPTIONS::ADF);
   
    testing.loadCSV("gamble.mat");    
   
    testing.evaluateSE(6);     

    arma::vec beta = testing.regression.getBeta();
    beta.print("beta:");
     
    testing.getStatistics(); 

    return 0;   
}

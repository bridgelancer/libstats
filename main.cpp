#include <iostream>
#include <armadillo>

#include "ADF/adf.h"

using namespace std;
using namespace arma;

int main()
{
   
    OPTIONS options;
    adf testing = adf(OPTIONS::ADF);
   
    testing.loadCSV("Training.csv");    
//    arma::vec y(10000);
//    y.randu();
    
//    testing.setObservation(y);
       
    testing.evaluateSE(17);     

    arma::vec beta = testing.regression.getBeta();
    beta.print("beta:");
     
    testing.getStatistics(); 

    return 0;   
}

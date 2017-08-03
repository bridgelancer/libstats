#include <iostream>
#include <armadillo>

#include "ADF/adf.h"

using namespace std;
using namespace arma;

int main()
{
    OPTIONS options;
    adf testing = adf(OPTIONS::DF);

    vec y = vec(4);
    y = ("0.1 0.2 0.3 0.6");

    testing.loadDesign("A.mat");
    testing.setObservation(y);
    
    testing.regression.getDesign();
    testing.regression.getObservation();
    testing.regression.evaluate();
    arma::vec beta = testing.regression.getBeta();

    beta.print("beta:");
    

    return 0;   
}

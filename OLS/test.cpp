#include <iostream>
#include <armadillo>

#include "ols.h"

using namespace std;
using namespace arma;

int main(){
    vec y = vec(4);
    y = ("0.1 0.2 0.3 0.6");
    
    ols testing = ols(arma::mat(), y);
    testing.getDesign();
    testing.getObservation();
    testing.getBeta();
      
        
    testing.loadDesign("A.mat");
    testing.setObservation(y);

    testing.evaluate();

    testing.getBeta();
    testing.saveBetaCSV();
}

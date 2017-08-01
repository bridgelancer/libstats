#include <iostream>
#include <armadillo>

#include "ols.h"

using namespace std;
using namespace arma;

int main(){
    ols testing = ols();
    testing.getDesign();
    testing.getObservation();
    testing.getBeta();
    
    mat A = mat(4,2);
    A = ("1 1; 1 2; 1 3; 1 4");
    
    vec y = vec(4);
    y = ("0.1 0.2 0.3 0.4");
    
    testing.setDesign(A);
    testing.setObservation(y);

    testing.evaluate();

    testing.getBeta();

}

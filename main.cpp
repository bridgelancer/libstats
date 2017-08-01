#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main()
{
    mat A = mat(); 
    vec x = vec();

    cout << x << endl;
       
    vec Y = vec(4);
    Y(0) = 0.1;
    Y(1) = 0.2;
    Y(2) = 0.3;
    Y(3) = 0.4;

    
    
    A = mat(4,2); // storing the regressor 1 an X for the four observations
    vec beta = vec(2);

    for (int i = 0; i < 4; i++){ 
        A(i,0) = 1;
        A(i,1) = i+1; 
    }
    
    beta = (A.t() * A).i() * A.t() * Y;    

    cout << beta;   
    
    return 0;
}

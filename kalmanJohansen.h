#include "sigpack.h"
#include "gsl/gsl_matrix.h"
#include <armadillo>

// need to include Johansen Library as well

namespace arma = armadillo;

class kalmanJohansen : public sp::KF
{   
public:
    kalmanJohansen();
    ~kalmanJohansen();

    void update(gsl_matrix* xMat_gsl);
    arma::mat getUpdate();

private:
    void gslToArma();

    gsl_matrix* xMat_gsl; // input from Johansen library
    arma::mat xMat;
}
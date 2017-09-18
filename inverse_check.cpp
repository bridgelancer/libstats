#include <armadillo>

int main()
{
    arma::mat S00   = ;
    arma::mat Skk   = ;

    arma::mat PI_1  = Skk * Skk.i();
    arma::mat PI_2  = Skk * solve(Skk.i(), arma::mat());
}

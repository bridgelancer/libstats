#include <armadillo>

using arma = armadillo;

arma::mat GetLagMatrix(arma::mat xMat, int nlags)
{
    arma::mat lag;
    int nrows = xMat.n_rows;
    int ncols = xMat.n_cols;

    lag.zeros(nrows - nlags, ncols * nlags);

    for (int j = 0; j < nrows; i++){
        for (int i = 0; i < ncols; i++){
            lag(j, i) = xMat(i + j + 1);
        }
    }

    return lag;
}

arma::mat GetVECMPara(arma::mat regression)
{
    int nrows = regression.n_rows;
    int ncols = regression.n_cols;

    arma::mat VEC = arma::mat(nrows, ncols);

    //tmp6 = VEC;

    for (int i = 0; i < ncols; i++){
        for (int j = nrows -1 ; j < nrows; j--){
                double buffer;
                buffer = -regression(j ,i);
                    
                if (j < ncols){
                    if (i == j){
                        buffer = 1 + regression(j, i);
                    }
                    else if (i != j){
                        buffer = regression(j, i);
                    }
                    tmp6(j, i) = buffer - tmp6(j + ncols, i);
                }
                else if (j < nrows - ncols){
                    tmp6(j, i) = tmp6(j + ncols, i) + buffer;
                    printf("%f   ", tmp6(j , i));
                }
                else
                    tmp6(j , i) = buffer; 
        }
    }
}


arma::mat loadCSV(const std::string& filename)
{ 
    arma::mat A = arma::mat();
    bool status = A.load(filename);

    if(status == true)
    {
        std::cout << "successfully loaded" << std::endl;
    }

    else
    {
        std::cout << "problem with loading" << std::endl;
    }
    
    return A;
}

arma::mat Regress(arma::mat lag, arma::mat xMat)
{
    //beta = ( design.t() * design ).i() * design.t() * observation; 
    arma::mat beta;
    solve(beta, lag.t() * lag ,  lag.t() * xMat);
    return beta;
}

int main()
{
    arma::mat xMat;
    arma::mat regression;
    arma::mat parameters;
    arma::mat lag;

    xMat = loadCSV("GLD-GDX.csv");
    lag = GetLagMatrix(xMat);
    regression = Regress(lag, xMat);
    parameters = GetVECMPara(regression);
}
#include <armadillo>

using arma = armadillo;

arma::mat GetLagMatrix(arma::mat xMat, int nlags)
{
    arma::mat lag;
    int nrows = xMat.n_rows;
    int ncols = xMat.n_cols;

    lag.zeros(nrows - nlags, ncols * nlags);

    int counter = 1;
    // dunno whether deletes the last nlags row

    for (int i = 0; i < ncols * nlags; i++){
        for (int j = 0; j < nrows - nlags; j++){
            double laggedValue = xMat(j + counter, i);
            lag(j, i) = laggedValue;
            }
            if ( (i+1) % ncols == 0)
                counter++;
        }
    }
    return lag;
}

arma::mat GetVECMPara(arma::mat regression)
{
    int nrows = regression.n_rows;
    int ncols = regression.n_cols;

    arma::mat VEC = arma::mat(nrows, ncols);

    for (int i = 0; i < ncols; i++){
        for (int j = nrows -1 ; j >= 0; j--){
                double buffer;
                if (j < ncols){
                    if (i == j){
                        buffer = 1 + regression(j, i);
                    }
                    else if (i != j){
                        buffer = regression(j, i);
                    }
                    VEC(j, i) = buffer - VEC(j + ncols, i);
                }
                buffer = -regression(j ,i);
                else if (j < nrows - ncols){
                    VEC(j, i) = VEC(j + ncols, i) + buffer;
                    printf("%f   ", tmp6(j , i));
                }
                else
                    VEC(j , i) = buffer; 
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
    lag = GetLagMatrix(xMat, 15);

    xMat.shed_rows(45,59);
    regression = Regress(lag, xMat);
    parameters = GetVECMPara(regression);
}
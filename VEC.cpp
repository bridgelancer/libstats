#include <armadillo>

using namespace arma;

arma::mat GetLagMatrix(arma::mat xMat, int nlags)
{
    int nrows = xMat.n_rows;
    int ncols = xMat.n_cols;

    arma::mat lag = arma::mat(nrows - nlags, ncols * nlags);

    int counter = 1;
    int counter2 = 0;
    // dunno whether deletes the last nlags row

    for (int i = 0; i < ncols * nlags; i++){
        for (int j = 0; j < nrows - nlags; j++){

            lag(j, i) = xMat(j + counter, counter2);

        }
            counter2++;
            if ( (i+1) % ncols == 0){
                counter++;
                counter2 = 0;
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
                else if (j < nrows - ncols){
                    buffer = -regression(j ,i);
                    VEC(j, i) = VEC(j + ncols, i) + buffer;
                }
                else{
                    buffer = -regression(j ,i);
                    VEC(j , i) = buffer;
                }
                printf("%.4f   ", VEC(j , i));
        }
        printf("\n\n\n");
    }
    return VEC;
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

void saveMatCSV(arma::mat Mat, std::string filename)
{ 
    Mat.save(filename, arma::csv_ascii);
}


int main()
{
    arma::mat xMat;
    arma::mat regression;
    arma::mat parameters;
    arma::mat lag;

    xMat = loadCSV("GLD-GDX.csv");

    lag = GetLagMatrix(xMat, 15);
    saveMatCSV(lag, "Lag.csv");

    xMat.shed_rows(45,59);
    regression = Regress(lag, xMat);
    saveMatCSV(regression, "Regression.csv");

    std::cout << "Testing \n";
    parameters = GetVECMPara(regression);
    saveMatCSV(parameters, "VECM_Para.csv");
}
#include <armadillo>

using namespace arma;


void saveMatCSV(arma::mat Mat, std::string filename)
{ 
    Mat.save(filename, arma::csv_ascii);
}

void saveMatCSV(arma::cx_mat Mat, std::string filename)
{ 
    Mat.save(filename, arma::csv_ascii);
}

arma::mat regress(arma::mat X, arma::mat Y)
{
    // beta = ( design.t() * design ).i() * design.t() * observation; 
    arma::mat beta;
    // solve(beta, X.t() * X ,  X.t() * Y);
    beta = (X.t() * X).i() * X.t() * Y;

    return beta;
}

// xMat should have the latest data at front, the last nlags # of observations will be discarded
arma::mat getLagMatrix(arma::mat xMat, int nlags)
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

arma::mat getVARPara(arma::mat xMat, arma::mat lag, int nlags)
{
    xMat.shed_rows(xMat.n_rows - nlags, xMat.n_rows - 1);
    saveMatCSV(xMat, "xMatShed.csv");
    arma::mat VARPara = regress(lag, xMat);
    return VARPara;
}

// The last one is wrong as well -> should be regression problem
arma::mat getVECMPara(arma::mat VARPara)
{
    int nrows = VARPara.n_rows;
    int ncols = VARPara.n_cols;

    arma::mat VEC = arma::mat(nrows, ncols);

    for (int i = 0; i < ncols; i++){
        for (int j = nrows -1 ; j >= 0; j--){
                double buffer;
                if (j < ncols){
                    if (i == j){
                        buffer = 1 - VARPara(j, i);
                    }

                    else if (i != j){
                        buffer = -VARPara(j, i);
                    }
                    VEC(j, i) = -(buffer + VEC(j + ncols, i));
                }
                else if (j < nrows - ncols){
                    buffer = -VARPara(j ,i);
                    VEC(j, i) = VEC(j + ncols, i) + buffer;
                }
                else{
                    buffer = -VARPara(j ,i);
                    VEC(j, i) = buffer;
                }
                printf("%.4f   ", VEC(j, i));
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
        std::cout << "Successfully loaded" << std::endl;
    }

    else
    {
        std::cout << "Problem with loading" << std::endl;
    }
    
    return A;
}


arma::mat getMatrixDiff(arma::mat xMat)
{
    int nrows = xMat.n_rows;
    int ncols = xMat.n_cols;

    arma::mat diff = arma::mat(nrows - 1, ncols);

    for (int i = 0; i < nrows -1; i++){
        for (int j = 0; j < ncols; j++){
            diff(i , j) = xMat(i , j) - xMat(i+1 , j);
        }
    }

    return diff;
}   


// handle the shedding within function, not checked
arma::mat getResidualDX(arma::mat xMat, arma::mat dLag, int nlags)
{
    arma::mat coeff;
    arma::mat dxMat;

    xMat.shed_rows(xMat.n_rows - nlags, xMat.n_rows -1); // regressing first differences of Xt against lag terms of first differences
    dxMat = getMatrixDiff(xMat);

    saveMatCSV(dxMat, "dxMat.csv");

    std::cout << "dxMat dimensions are " << dxMat.n_rows << "   " << dxMat.n_cols << std::endl;
    std::cout << "dLag dimensions are " << dLag.n_rows << "   " << dLag.n_cols << std::endl;
    printf("\n");

    coeff = regress(dLag, dxMat);
    saveMatCSV(coeff, "coeffDX.csv");

    return (dxMat - dLag * coeff);
}

arma::mat getResidualX(arma::mat xMat, arma::mat dLag, int nlags)
{
    arma::mat coeff;
    
    xMat.shed_rows(0, nlags); // regressing Xt-k against lag terms of first differences, deleting first nlags + 1 rows

    std::cout << "xMat dimensions are " << xMat.n_rows << "   " << xMat.n_cols << std::endl;
    std::cout << "dLag dimensions are " << dLag.n_rows << "   " << dLag.n_cols << std::endl;
    
    printf("\n");
    
    saveMatCSV(xMat, "xMat.csv");

    coeff = regress(dLag, xMat);
    saveMatCSV(coeff, "coeffX.csv");

    arma::mat residualX = xMat - dLag * coeff;
    residualX.raw_print(std::cout , "resdwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwiualX:");
    return (xMat - dLag * coeff);
}

arma::mat getEigenInput(arma::mat residualDX, arma::mat residualX)
{
    arma::mat Skk;
    arma::mat Sk0;
    arma::mat S0k;
    arma::mat S00;

    Skk = 1.0/((double)residualX.n_rows) * residualX.t() * residualX;
    Sk0 = 1.0/((double)residualDX.n_rows) * residualDX.t() * residualX;
    S0k = 1.0/((double)residualX.n_rows) * residualX.t() * residualDX;
    S00 = 1.0/((double)residualDX.n_rows) * residualDX.t() * residualDX;

    std::cout << "Skk dimensions are " << Skk.n_rows << "   " << Skk.n_cols << std::endl;
    std::cout << "Sk0 dimensions are " << Sk0.n_rows << "   " << Sk0.n_cols << std::endl;

    arma::mat One;

    Skk.raw_print(std::cout, "Skk:");
    Sk0.raw_print(std::cout, "Sk0:");
    S0k.raw_print(std::cout, "S0k:");
    S00.raw_print(std::cout, "S00:");

    arma::mat SkkInv = solve(Skk, eye<mat>(Skk.n_rows, Skk.n_rows));
    arma::mat S00Inv = solve(S00, eye<mat>(S00.n_rows, S00.n_rows));
    arma::mat eigenInput = Skk.i() * (Sk0 * S00.i() * S0k);
    return eigenInput;
}

arma::cx_mat getEigenOutput(arma::mat eigenInput)
{
    arma::cx_vec eigval;
    arma::cx_mat eigvec;

    eig_gen(eigval, eigvec, eigenInput);
    return eigvec;
}

arma::cx_vec getEigenVal(arma::mat eigenInput)
{
    arma::cx_vec eigval;
    arma::cx_mat eigvec;

    eig_gen(eigval, eigvec, eigenInput);
    return eigval;
}

int main()
{
    arma::mat xMat;
    arma::mat dxMat;
    arma::mat VARPara;
    arma::mat VECPara;
    arma::mat lag;
    arma::mat dLag;
    arma::mat residualDX;
    arma::mat residualX;
    arma::mat eigenInput;
    arma::cx_mat eigvec;
    arma::cx_vec eigval;

    xMat = loadCSV("GLD-GDX.csv");
    saveMatCSV(xMat, "xMat60.csv");

    lag = getLagMatrix(xMat, 15);
    saveMatCSV(lag, "Lag.csv");
    dLag = getMatrixDiff(lag);
    saveMatCSV(dLag, "dLag.csv");

    saveMatCSV(lag, "Lag.csv");

    VARPara = getVARPara(xMat, lag, 15);
    saveMatCSV(VARPara, "VAR_Para.csv");

    std::cout << "Testing \n";
    VECPara = getVECMPara(VARPara).t();
    saveMatCSV(VECPara, "VECM_Para.csv");

    // okay for now, working on below

    residualX = getResidualX(xMat, dLag, 15);
    residualDX = getResidualDX(xMat, dLag, 15);
    saveMatCSV(residualX, "residualX.csv");
    saveMatCSV(residualDX, "residualDX.csv");
    eigenInput = getEigenInput(residualDX, residualX);
    saveMatCSV(eigenInput, "Eigeninput.csv");

    eigvec = getEigenOutput(eigenInput);
    saveMatCSV(eigvec, "Eigenvec.csv");

    eigval = getEigenVal(eigenInput);
    saveMatCSV(eigval, "Eigenval.csv");

    // getResidualDX/X
    // getEigenInput
    // Solve the Eigen matrix for eigenvalues/vectors
    // Perform the statistics test
}
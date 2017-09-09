#include <armadillo>
#include <fstream>
#include <iomanip>
#include <cmath>

// @TODO change function signature to VEC::function();
using namespace arma;

// Do not commit until FGLS is fixed

// by default, the output precision is up to 4 dp
// if the precision is changed to nPrecision dp, pls change the term (4 + 3/2) to (nPrecision + 3/2) respectively;

// can use raw_print instead?
void saveMatCSV(arma::mat Mat, std::string filename)
{ 
    std::ofstream stream = std::ofstream();
    stream.open(filename, std::ofstream::out | std::ofstream::trunc);

    int nrows = Mat.n_rows;
    int ncols = Mat.n_cols;

    stream << std::setprecision(4);
    stream.setf( std::ios::fixed, std:: ios::floatfield );
    
    arma::vec maxVal = arma::vec(ncols);
    arma::vec minVal = arma::vec(ncols);
    arma::vec status = arma::vec(ncols);

    for (int j = 0; j < ncols; j++){
        maxVal(j) = max(arma::abs(Mat.col(j))); // finding the value of element in each column with largest magnitude
        minVal(j) = min(Mat.col(j)); // finding the smallest value

        if (maxVal(j) != std::abs(minVal(j))){ // if the value of largest magnitude is not the one of smallest value (i.e. +100/ -0.1)

            if ( maxVal(j) < 1 && std::abs(minVal(j)) < 1 && minVal(j) < 0.0)
                status(j) = 0;
            else if ( maxVal(j) < 1 && std::abs(minVal(j)) < 1 && minVal(j) > 0.0)
                status(j) = 1;
            else{
                double tmpMaxVal = maxVal(j);
                double tmpMinVal = std::abs(minVal(j));
                if ( maxVal(j) < 1 )
                     tmpMaxVal = maxVal(j) + 1;
                if ( std::abs(minVal(j)) < 1 )
                    tmpMinVal = std::abs(minVal(j)) + 1;

                if( (int)(log10(tmpMaxVal)) - (int)(log10(tmpMinVal)) < 1.0 && minVal(j) < 0.0)
                    status(j) = 0;
                else
                    status(j) = 1;
            }
        } 
        else // the value of largest magnitude is the one of smallest value (i.e. +10/ -100)
            status(j) = 0;
    }

    for (int i = 0; i < nrows; i++){
        for (int j = 0; j < ncols; j++){
            if (maxVal(j) > 1){
                if ( status(j) == 0 )
                    stream << std::setfill(' ') << std::setw( (int)log10(maxVal(j)) + (4 + 3) ) << Mat(i, j); // setting width, extra 3 spaces are added for storing ".", negative sign, and (int)log(x) rounds down;
                else if ( status(j) == 1 )
                    stream << std::setfill(' ') << std::setw( (int)log10(maxVal(j)) + (4 + 2) ) << Mat(i, j); // setting width, extra 2 spaces are added for storing ".", and (int)log(x) rounds down;
            }
            else {
                if ( status(j) == 0 )
                    stream << std::setfill(' ') << std::setw( (int)log10(1 + maxVal(j)) + (4 + 3) ) << Mat(i, j); // setting width, extra 3 spaces are added for storing ".", negative sign, and (int)log(x) rounds down;
                else if ( status(j) == 1 )
                    stream << std::setfill(' ') << std::setw( (int)log10(1 + maxVal(j)) + (4 + 2) ) << Mat(i, j); // setting width, extra 2 spaces are added for storing ".", and (int)log(x) rounds down;
            }

            if (j != ncols - 1)
                stream << ", ";
        }
        stream << "\n";
    }
    stream.close();
}

void saveMatCSV(arma::cx_mat Mat, std::string filename)
{ 
    Mat.save(filename, arma::csv_ascii);
}

arma::mat regressOLS(arma::mat X, arma::mat Y)
{
    // beta = (X.t() * X).i() * X.t() * Y; 
    arma::mat beta;
    solve(beta, X.t() * X ,  X.t() * Y);

    return beta;
}

arma::mat regressGLS(arma::mat X, arma::mat Y, arma::mat covariance)
{
    arma::mat beta;

    arma::mat covarianceI;
    solve(covarianceI, covariance, eye(size(covariance)));

    solve(beta, X.t() * covarianceI * X, X.t() * covarianceI * Y);

    return beta;
}
 
// @TODO
arma::mat getCovarianceMatrix(arma::mat beta, arma::mat xMat, arma::mat lag)
{
    arma::mat error = arma::mat(xMat.n_rows, xMat.n_cols);
    xMat.shed_rows(xMat.n_rows - lag.n_cols/xMat.n_cols, xMat.n_rows - 1);
    
    error = xMat - lag * beta;

    arma::mat covariance;
    covariance = 1.0/(double) (xMat.n_rows - lag.n_cols/xMat.n_cols) * error * error.t();
    // (xMat - Z * beta) * (xMat.t() - beta.t() * Z.t())
    return covariance;
 // assume independent and error term not skewed. the current calculated magnitude of staistics would be greater than intended.
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

    // new things - add two rows of 1 in front of the lag
    arma::mat B = arma::ones<mat>(nrows - nlags, ncols);
    lag = join_rows(B, lag);

    
    return lag;
}

arma::mat getBeta(arma::mat xMat, arma::mat lag, int nlags)
{
    // @TODO need to sort out the matrix multiplication error
    xMat.shed_rows(xMat.n_rows - (lag.n_cols - 2)/xMat.n_cols, xMat.n_rows - 1);
    saveMatCSV(xMat, "xMatShed.csv");
    arma::mat beta = regressOLS(lag, xMat);
    return beta;
}

arma::mat getVARPara(arma::mat xMat, arma::mat lag, arma::mat covariance)
{
    // @TODO need to sort out the matrix multiplication error
    xMat.shed_rows(xMat.n_rows - (lag.n_cols - 2)/xMat.n_cols, xMat.n_rows - 1);
    arma::mat VARPara = regressGLS(lag, xMat, covariance);
    return VARPara;
}

// The last row is wrong as well -> should be regression problem
arma::mat getVECMPara(arma::mat VARPara)
{
    int nrows = VARPara.n_rows;
    int ncols = VARPara.n_cols;

    arma::mat VEC = arma::mat(nrows, ncols);

    for (int i = 0; i < ncols; i++){
        for (int j = nrows -1 ; j >= 0; j--){
                double buffer;
                if (j < ncols){
                    VEC(j, i) = VARPara(j, i);
                }
                else if (j < ncols* 2){
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
        }
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

arma::mat demean(arma::mat X)
{ 
    arma::mat mean = arma::mean(X, 0); // finding the average value for each col

    // check validity
    arma::mat demean;

    demean = arma::mat(size(X));
    for (int i = 0; i < X.n_cols; i++){
        demean.col(i).fill(demean(i));
    }

    demean = X - demean; 

    return demean;
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

    coeff = regressOLS(dLag, dxMat);
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

    coeff = regressOLS(dLag, xMat);
    saveMatCSV(coeff, "coeffX.csv");

    arma::mat residualX = xMat - dLag * coeff;
    return (xMat - dLag * coeff);
}

arma::mat getEigenInput(arma::mat residualDX, arma::mat residualX)
{
    arma::mat Skk;
    arma::mat Sk0;
    arma::mat S0k;
    arma::mat S00;

    // @TODO refer to definition of Sk0/S0k
    Skk = 1.0/((double)residualX.n_rows) * residualX.t() * residualX;
    Sk0 = 1.0/((double)residualX.n_rows) * residualX.t() * residualDX;
    S0k = 1.0/((double)residualDX.n_rows) * residualDX.t() * residualX;
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

    eigenInput.raw_print(std::cout, "eigenInput");
    return eigenInput;
}

// @TODO combine the getEigenOuput and getEigen Val 
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
    arma::mat beta;
    arma::mat covariance;
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

    lag = getLagMatrix(xMat, 17);
    saveMatCSV(lag, "Lag.csv");
    dLag = getMatrixDiff(lag);
    saveMatCSV(dLag, "dLag.csv");

    beta = getBeta(xMat, lag, 17);
    saveMatCSV(beta, "beta.csv");

    /* now broken due to the ones
    covariance = getCovarianceMatrix(beta, xMat, lag);
    saveMatCSV(covariance, "covariance.csv");

    arma::mat covarianceI;
    solve(covarianceI, covariance, eye(size(covariance)));
    saveMatCSV(covarianceI, "covarianceI.csv");

    arma::mat I = covariance * covarianceI;
    saveMatCSV(I, "I.csv");
    */
    VARPara = getVARPara(xMat, lag, eye(size(xMat.n_rows - (lag.n_cols - 2)/xMat.n_cols, xMat.n_rows - (lag.n_cols - 2)/xMat.n_cols)));
    saveMatCSV(VARPara, "VAR_Para.csv");
    VECPara = getVECMPara(VARPara).t();
    saveMatCSV(VECPara, "VECM_Para.csv");

    // okay for now, working on below

    /*

    residualX = getResidualX(xMat, dLag, 17);
    residualDX = getResidualDX(xMat, dLag, 17);
    saveMatCSV(residualX, "residualX.csv");
    saveMatCSV(residualDX, "residualDX.csv");
    eigenInput = getEigenInput(residualDX, residualX);
    saveMatCSV(eigenInput, "Eigeninput.csv");

    eigvec = getEigenOutput(eigenInput);
    saveMatCSV(eigvec, "Eigenvec.csv");

    eigval = getEigenVal(eigenInput);
    saveMatCSV(eigval, "Eigenval.csv");
    */

    // Perform the statistics test
}
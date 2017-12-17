#include <armadillo>
#include <fstream>
#include <iomanip>
#include <cmath>

// This class has already been implemented to a relevant class structure "VECM.h" and its corresponding .cpp files
// This class was verified against the outcome of cajo.R using the dataset GLD-GDX.csv attached with this repository

// The file stands as a stand alone, development version of VECM.h/VEM.cpp, with suitable main function coded for testing
// Warning: Executing this file produces numerous .csv outputs (mainly for checking purposes)

using namespace arma;

// Do not commit until FGLS is implemented

// Critical values of eigenvalues of Johansen Test
struct cVals
{
    arma::mat eigen = { {6.5, 8.18, 11.65},
                        {12.91, 14.90, 19.19},
                        {18.9, 21.07, 25.75},
                        {24.78, 27.14, 32.14},
                        {30.84, 33.32, 38.78},
                        {36.25, 39.43, 44.59},
                        {42.06, 44.91, 51.30},
                        {48.43, 51.07, 57.07},
                        {54.01, 57.00, 63.37},
                        {59.00, 62.42, 68.61},
                        {65.07, 68.27, 74.36}, };
};

// Code modified from some sources online for implmeenting Johansen test
// See pivoted cholesky process for detailes
arma::mat pivoted_cholesky(const arma::mat & A, double eps, arma::uvec & pivot) {
    if(A.n_rows != A.n_cols)
      throw std::runtime_error("Pivoted Cholesky requires a square matrix!\n");
  
    // Returned matrix
    arma::mat L;
    L.zeros(A.n_rows,A.n_cols);
  
    // Loop index
    size_t m(0);
    // Diagonal element vector
    arma::vec d(arma::diagvec(A)); //d = {A(0,0), A(1,1)}; in column
    // Error
    double error(arma::max(d));
  
    // Pivot index
    arma::uvec pi(arma::linspace<arma::uvec>(0,d.n_elem-1,d.n_elem));  //d.n_elem = 2
    //generat equal spaced uvec pi of 0, 1, ..., d.n_elem-1
  
    while(error>eps && m<d.n_elem) {
      // Errors in pivoted order
      arma::vec errs(d(pi));
      // Sort the upcoming errors so that largest one is first
      arma::uvec idx=arma::stable_sort_index(errs.subvec(m,d.n_elem-1),"descend");
  
      // Update the pivot index
      arma::uvec pisub(pi.subvec(m,d.n_elem-1));
      pisub=pisub(idx);
      pi.subvec(m,d.n_elem-1)=pisub;
  
      // Pivot index
      size_t pim=pi(m);
      //printf("Pivot index is %4i with error %e, error is %e\n",(int) pim, d(pim), error);
  
      // Compute diagonal element
      L(m,pim)=sqrt(d(pim));
  
      // Off-diagonal elements
      for(size_t i=m+1;i<d.n_elem;i++) {
        size_t pii=pi(i);
        // Compute element
        L(m,pii)= (m>0) ? (A(pim,pii) - arma::dot(L.col(pim).subvec(0,m-1),L.col(pii).subvec(0,m-1)))/L(m,pim) : (A(pim,pii))/L(m,pim);
        // Update d
        d(pii)-=L(m,pii)*L(m,pii);
      }
  
      // Update error
      error=arma::max(d(pi.subvec(m,pi.n_elem-1))); //second update, subvec gg

      //diagonal matrix's element(pi.subvec(m+1, pi.n_elem-1)))

      // Increase m
      m++;
    }
    //printf("Final error is %e\n",error);

    // Transpose to get Cholesky vectors as columns
    arma::inplace_trans(L);
  
    // Drop unnecessary columns
    if(m<L.n_cols)
      L.shed_cols(m,L.n_cols-1);
  
    // Store pivot
    pivot=pi.subvec(0,m-1);
  
    return L;
}

// By default, the output precision is up to 4 dp
// If the precision is changed to nPrecision dp, pls change the term (4 + 3/2) to (nPrecision + 3/2) respectively;

// @TODO Consider using rawprint instead of direct manipulation of stream
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

// Saving comple matrix arma::cx_mat Mat
void saveMatCSV(arma::cx_mat Mat, std::string filename)
{ 
    Mat.save(filename, arma::csv_ascii);
}

// For convenience, the class implementation should utilize the method derived from the OLS class instead
arma::mat regressOLS(arma::mat X, arma::mat Y)
{
    // beta = (X.t() * X).i() * X.t() * Y; 
    arma::mat beta;
    solve(beta, X.t() * X ,  X.t() * Y);

    return beta;
}

// @TODO - This is currently not working, dependent on function getCovarianceMatrix
arma::mat regressGLS(arma::mat X, arma::mat Y, arma::mat covariance)
{
    arma::mat beta;

    arma::mat covarianceI;
    solve(covarianceI, covariance, eye(size(covariance))); // solving for inverse of covariance

    solve(beta, X.t() * covarianceI * X, X.t() * covarianceI * Y);

    return beta;
}
 
// @TODO - FGLS-related feature, not working at the moment
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
// getLagMatrix produce the lagged series of the original xMat data
arma::mat getLagMatrix(arma::mat xMat, int nlags)
{
    int nrows = xMat.n_rows;
    int ncols = xMat.n_cols;

    arma::mat lag = arma::mat(nrows - nlags + 1, ncols * nlags);

    int counter = 0;
    int counter2 = 0;
    // dunno whether deletes the last nlags row

    for (int i = 0; i < ncols * nlags; i++){
        for (int j = 0; j < nrows - nlags + 1; j++){
                lag(j, i) = xMat(j + counter, counter2);
        }
            counter2++;
            if ( (i+1) % ncols == 0){
                counter++;
                counter2 = 0;
            }
    }

    // add one rows of 1 behind lag matrix
    arma::mat B = arma::ones<mat>(nrows - nlags + 1, 1);
    lag = join_rows(lag, B);
    
    return lag;
}


arma::mat getBeta(arma::mat xMat, arma::mat lag, int nlags)
{
    lag.shed_row(0);
    xMat.shed_rows(xMat.n_rows - (lag.n_cols - 1)/xMat.n_cols, xMat.n_rows - 1);
    saveMatCSV(xMat, "xMatShed.csv");
    arma::mat beta = regressOLS(lag, xMat);
    return beta;
}

arma::mat getVARPara(arma::mat xMat, arma::mat lag, arma::mat covariance)
{
    lag.shed_row(0);
    xMat.shed_rows(xMat.n_rows - (lag.n_cols - 1)/xMat.n_cols, xMat.n_rows - 1);
    arma::mat VARPara = regressGLS(lag, xMat, covariance); //regressGLS currently not working
    
    return VARPara;
}

// extra two cols in front
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
                VEC(j, i) = (buffer + VEC(j + ncols, i));
            }
            else if (j < nrows - ncols - 1){
                buffer = -VARPara(j ,i);
                VEC(j, i) = VEC(j + ncols, i) + buffer;
            }
            else{
                buffer = -VARPara(j ,i);
                VEC(j, i) = buffer;
            }
        }
    }

    VEC.shed_rows(0, VARPara.n_cols-1);
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

// may consider giving back the first row
// getMatrixDiff subtract the newer data from the lagged data
// @TODO Consider reimplementeing the function with standard armadillo methods
arma::mat getMatrixDiff(arma::mat lag, arma::mat xMat)
{
    int nrows = lag.n_rows;
    int ncols = lag.n_cols;

    arma::mat diff = arma::mat(nrows - 1, ncols);

    for (int i = 0; i < nrows - 1; i++){
        for (int j = 0; j < ncols; j++){
            diff(i , j) = lag(i+1 , j) - lag(i , j);
        }
    }

    return diff;
}   

// not in use for the moment
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

arma::mat getEigenInput(arma::mat xMat, arma::mat dLag, int nlags) // xMat = x, dLag = Z
{
    dLag.shed_col(dLag.n_cols - 1);

    int P = xMat.n_cols;
    int N = xMat.n_rows;

    arma::mat Z0;
    Z0 = dLag.cols(0 , P - 1);
    /***
    Z <- embed(diff(x), K) <=> dLag
    Z0 <- Z[, 1:P] #Z0 only picking the first # of stocks of cols, i.e the first difference data only
    ***/
    arma::mat Z1; 
    Z1 = dLag.cols(P, dLag.n_cols - 1);
    arma::mat B = arma::ones<mat>(Z1.n_rows, 1);

    Z1 = join_rows(B, Z1);
    // Z1 <- Z[, -c(1:P)] shed the first P cols
    // Z1 <- cbind(1, Z1) # Z1
    arma::mat Zk; 
    Zk = xMat.rows(1, N - nlags); 
    // ZK <- x[-N, ][K:(N - 1), ] # Zk

    // R is one row more than C++
    saveMatCSV(Z0, "Z0.csv"); // R -> first row extra
    saveMatCSV(Z1, "Z1.csv"); // R -> bottom row extra
    saveMatCSV(Zk, "Zk.csv"); // R -> first row extra
    std::cout << "checking";

    int n = Z0.n_rows;

    arma::mat M00 = Z0.t() * Z0 / n;
    arma::mat M11 = Z1.t() * Z1 / n;
    arma::mat Mkk = Zk.t() * Zk / n;
    arma::mat M01 = Z0.t() * Z1 / n;
    arma::mat M0k = Z0.t() * Zk / n;
    arma::mat Mk0 = Zk.t() * Z0 / n;
    arma::mat M10 = Z1.t() * Z0 / n;
    arma::mat M1k = Z1.t() * Zk / n;
    arma::mat Mk1 = Zk.t() * Z1 / n;
    arma::mat M11inv = arma::solve(M11, arma::eye<mat>(size(M11)));

    arma::mat R0 = Z0 - (M01 * M11inv * Z1.t()).t();
    arma::mat Rk = Zk - (Mk1 * M11inv * Z1.t()).t();

    arma::mat S00 = M00 - M01 * M11inv * M10;
    arma::mat S0k = M0k - M01 * M11inv * M1k;
    arma::mat Sk0 = Mk0 - Mk1 * M11inv * M10;
    arma::mat Skk = Mkk - Mk1 * M11inv * M1k;

    Skk.raw_print(std::cout, "Skk:");
    Sk0.raw_print(std::cout, "Sk0:");
    S0k.raw_print(std::cout, "S0k:");
    S00.raw_print(std::cout, "S00:");

    arma::mat SkkInv = solve(Skk, eye<mat>(size(Skk)));
    arma::mat S00Inv = solve(S00, eye<mat>(S00.n_rows, S00.n_rows));

    arma::mat I = Skk * SkkInv;
    saveMatCSV(I, "I.csv");
   
    arma::mat PI = S0k * Skk.i();
    PI.raw_print(std::cout, "PI:");

    arma::uvec pivot; // could delete if not needed

    arma::mat C = pivoted_cholesky(Skk, 0.01, pivot);
    saveMatCSV(C, "C.csv");

    arma::mat eigenInput = C.i() * (Sk0 * S00Inv * S0k) * C.i().t();
    eigenInput.raw_print(std::cout, "eigenInput");

    return eigenInput;
}

// @TODO combine the getEigenOuput and getEigen Val 
arma::cx_mat getEigenOutput(arma::mat eigenInput)
{
    arma::cx_vec eigval;
    arma::cx_mat eigvec;

    eig_gen(eigval, eigvec, eigenInput);
    return eigvec; // equivalent to e
}

arma::cx_vec getEigenVal(arma::mat eigenInput)
{
    arma::cx_vec eigval;
    arma::cx_mat eigvec;

    eig_gen(eigval, eigvec, eigenInput);
    return eigval; // equivalent to lambda
}

arma::mat getVorg(arma::mat C, arma::cx_mat eigvec)
{
    arma::mat real = arma::real(eigvec);
    arma::mat Vorg = (solve(C, eye(size(C)))).t() * real;
    return Vorg;
}

arma::mat getStatistics(arma::cx_vec eigval, arma::mat dLag){
    double N = dLag.n_rows;

    arma::mat stats;
    arma::mat eigen = arma::real(eigval);

    arma::mat one = arma::ones<mat>(size(eigen));
    arma::mat n = arma::mat(size(eigen));
    n.fill(-N);

    stats = n % log(one - eigen); // already negative

    return stats;
}


// refer to the structure of ca.jo@cvals, 1 = significant, 0 = insignficant
arma::mat getTest(arma::mat stats, arma::mat xMat, cVals c)
{
    arma::mat cVal = c.eigen;
    int K = xMat.n_cols;
    
    arma::mat test = arma::mat(K, 3);
    for (int r = 0; r < K; r++){
        for (int c = 0; c< 3; c++){
            if (cVal(r,c) < stats(r))
                test(r,c) = 1;
            else
                test(r,c) = 0;
        }
    }

    return test;
}

// The main function is largely implemented in the class structure of "VECM.compute"
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
    arma::mat Vorg;
    arma::mat V;
    arma::mat C;
    arma::mat stats;
    arma::mat test;
    

    xMat = loadCSV("GLD-GDX.csv");
    saveMatCSV(xMat, "xMat60.csv");

    lag = getLagMatrix(xMat, 16);
    saveMatCSV(lag, "Lag.csv");
    dLag = getMatrixDiff(lag, xMat);
    saveMatCSV(dLag, "dLag.csv");

    beta = getBeta(xMat, lag, 16);
    saveMatCSV(beta, "beta.csv");

    /***Possibly for FGLS to match MATLAB, not attempting for the moment

    covariance = getCovarianceMatrix(beta, xMat, lag);
    saveMatCSV(covariance, "covariance.csv");

    arma::mat covarianceI;
    solve(covarianceI, covariance, eye(size(covariance)));
    saveMatCSV(covarianceI, "covarianceI.csv");

    arma::mat I = covariance * covarianceI;
    saveMatCSV(I, "I.csv");
    ******************************************************************/

    VARPara = getVARPara(xMat, lag, eye(size(xMat.n_rows - (lag.n_cols - 1)/xMat.n_cols, xMat.n_rows - (lag.n_cols - 1)/xMat.n_cols)));
    saveMatCSV(VARPara, "VAR_Para.csv"); 
    VECPara = getVECMPara(VARPara);
    saveMatCSV(VECPara, "VECM_Para.csv"); // GAMMA

    eigenInput = getEigenInput(xMat, dLag, 16);
    saveMatCSV(eigenInput, "Eigeninput.csv");

    eigvec = getEigenOutput(eigenInput); // e
    saveMatCSV(eigvec, "Eigenvec.csv");

    eigval = getEigenVal(eigenInput); // lambda
    saveMatCSV(eigval, "Eigenval.csv");

    C = loadCSV("C.csv");
    Vorg = getVorg(C, eigvec);
    saveMatCSV(Vorg, "Vorg.csv"); // to 3dp accuracy

    // Perform the statistics test
    stats = getStatistics(eigval, dLag);
    saveMatCSV(stats, "stats.csv");
    
    cVals c = cVals();
    test = getTest(stats, xMat, c);
    saveMatCSV(test, "results.csv");
}
#include <armadillo>
#include <fstream>
#include <iomanip>
#include <cmath>

#include "VECM.h"

arma::mat VECM::crit_eigen = { {6.5, 8.18, 11.65},
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

arma::mat pivoted_cholesky(const arma::mat & A, double eps, arma::uvec & pivot)
{
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
// by default, the output precision is up to 4 dp
// if the precision is changed to nPrecision dp, pls change the term (4 + 3/2) to (nPrecision + 3/2) respectively;
VECM::VECM():
    _observation(),
    _lag(0),
    _test_stat(),
    _VARPara(),
    _Gamma(),
    _Pi(),
    _C(),
    _eigvec(),
    _eigval(),
    _Vorg(),
    _eigenInput(),
    _covariance(),
    _beta(),
    _lag_matrix(),
    _d_lag_matrix(),
    _Z0(),
    _Z1(),
    _ZK()
{}

VECM::VECM(arma::mat observation):
    _observation(observation),
    _lag(0),
    _test_stat(),
    _VARPara(),
    _Gamma(),
    _Pi(),
    _C(),
    _eigvec(),
    _eigval(),
    _Vorg(),
    _eigenInput(),
    _covariance(),
    _beta(),
    _lag_matrix(),
    _d_lag_matrix(),
    _Z0(),
    _Z1(),
    _ZK()
{}

VECM::~VECM()
{}

void VECM::compute(int nlags)
{
    arma::mat stats;

    _lag = nlags;
    _lag_matrix = computeLagMatrix();
    _d_lag_matrix = getMatrixDiff();
    _beta = computeBeta();

    /***Possibly for FGLS to match MATLAB, not attempting for the moment

    _covariance = computeCovarianceMatrix(beta, _observation, _lag_matrix);
    saveMatCSV(_covariance, "covariance.csv");

    arma::mat _covarianceI;
    solve(_covarianceI, _covariance, eye(size(_covariance)));
    saveMatCSV(_covarianceI, "covarianceI.csv");

    arma::mat I = _covariance * _covarianceI;
    saveMatCSV(I, "I.csv");
    ******************************************************************/

    _VARPara = computeVARPara();
    _Gamma = computeGamma();

    getEigenInput();
    getEigenOutput(); // e
    getVorg();

    // Perform the statistics _test_stat
    stats = getStatistics();
    _test_stat = getTest(stats);
}

void VECM::saveMatCSV(arma::mat Mat, std::string filename)
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

void VECM::saveMatCSV(arma::cx_mat Mat, std::string filename)
{
    Mat.save(filename, arma::csv_ascii);
}

// @TODO - might not need this?
arma::mat VECM::computeCovarianceMatrix()
{
    arma::mat error = arma::mat(_observation.n_rows, _observation.n_cols);
    _observation.shed_rows(_observation.n_rows - _lag_matrix.n_cols/_observation.n_cols, _observation.n_rows - 0);

    error = _observation - _lag_matrix * _beta;

    arma::mat _covariance;
    _covariance = 1.0/(double) (_observation.n_rows - _lag_matrix.n_cols/_observation.n_cols) * error * error.t();
    // (xMat - Z * _beta) * (xMat.t() - _beta.t() * Z.t())
    return _covariance;
 // assume independent and error term not skewed. the current calculated magnitude of staistics would be greater than intended.
}


// _observation should have the latest data at front, the last _lag # of observations will be discarded
arma::mat VECM::computeLagMatrix()
{
    int nrows = _observation.n_rows;
    int ncols = _observation.n_cols;

    arma::mat _lag_matrix = arma::mat(nrows - _lag + 1, ncols * _lag);

    int counter = 0;
    int counter2 = 0;
    // dunno whether deletes the last nlags row

    for (int c = 0; c < ncols * _lag; c++){
        for (int r = 0; r < nrows - _lag + 1; r++){
                _lag_matrix(r, c) = _observation(r + counter, counter2);
        }
            counter2++;
            if ( (c+1) % ncols == 0){
                counter++;
                counter2 = 0;
            }
    }

    // add one rows of 1 behind _lag_matrix matrix
    arma::mat B = arma::ones<arma::mat>(nrows - _lag + 1, 1);
    _lag_matrix = join_rows(_lag_matrix, B);

    return _lag_matrix;
}

arma::mat VECM::computeBeta()
{
    auto lag_matrix = _lag_matrix;
    auto observation = _observation;
    lag_matrix.shed_row(0);
    observation.shed_rows(observation.n_rows - (lag_matrix.n_cols - 1)/observation.n_cols, observation.n_rows - 1);
    arma::mat beta = regressOLS(lag_matrix, observation);
    return beta;
}

arma::mat VECM::computeVARPara()
{
    auto lag_matrix = _lag_matrix;
    auto observation = _observation;
    lag_matrix.shed_row(0);
    observation.shed_rows(observation.n_rows - (lag_matrix.n_cols - 1)/observation.n_cols, observation.n_rows - 1);
    arma::mat VARPara = regressOLS(lag_matrix, observation);

    return VARPara;
}

// extra two cols in front
arma::mat VECM::computeGamma()
{
    int nrows = _VARPara.n_rows;
    int ncols = _VARPara.n_cols;

    arma::mat VEC = arma::mat(nrows, ncols);

    for (int i = 0; i < ncols; i++){
        for (int j = nrows -1 ; j >= 0; j--){
            double buffer;
            if (j < ncols){
                VEC(j, i) = _VARPara(j, i);
            }
            else if (j < ncols* 2){
                if (i == j){
                    buffer = 1 - _VARPara(j, i);
                }

                else if (i != j){
                    buffer = -_VARPara(j, i);
                }
                VEC(j, i) = (buffer + VEC(j + ncols, i));
            }
            else if (j < nrows - ncols - 1){
                buffer = -_VARPara(j ,i);
                VEC(j, i) = VEC(j + ncols, i) + buffer;
            }
            else{
                buffer = -_VARPara(j ,i);
                VEC(j, i) = buffer;
            }
        }
    }

    //for (int c = 0; c < ncols; c++){
        //for (int r = nrows -1 ; r >= 0; r--){
            //double buffer;
            //if (r < ncols){
                //VEC(r, c) = _VARPara(r, c);
            //}
            //else if (r < ncols* 2){
                //if (r == c){
                    //buffer = 1 - _VARPara(r, c);
                //}
//
                //else if (r != c){
                    //buffer = -_VARPara(r, c);
                //}
                //VEC(r, c) = (buffer + VEC(r + ncols, c));
            //}
            //else if (r < nrows - ncols - 1){
                //buffer = -_VARPara(r ,c);
                //VEC(r, c) = VEC(r + ncols, c) + buffer;
            //}
            //else{
                //buffer = -_VARPara(r ,c);
                //VEC(r, c) = buffer;
            //}
        //}
    //}

    VEC.shed_rows(0, _VARPara.n_cols-1);
    return VEC;
}

// load data to _observation
arma::mat VECM::loadCSV(const std::string& filename)
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

    _observation = A;

    return A;
}

// may consider adding back the first row
arma::mat VECM::getMatrixDiff()
{
    int nrows = _lag_matrix.n_rows;
    int ncols = _lag_matrix.n_cols;

    arma::mat diff = arma::mat(nrows - 1, ncols);

    for (int r = 0; r < nrows - 1; r++){
        for (int c = 0; c < ncols; c++){
                diff(r , c) = _lag_matrix( r+1 , c) - _lag_matrix(r , c);
        }
    }

    return diff;
}

// not in use for the moment
arma::mat VECM::demean(arma::mat X)
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

void VECM::getEigenInput() // _observation = x, _d_lag_matrix = Z
{
    _d_lag_matrix.shed_col(_d_lag_matrix.n_cols - 1);

    int P = _observation.n_cols;
    int N = _observation.n_rows;

    arma::mat Z0 = _d_lag_matrix.cols(0 , P - 1);
    /***
    Z <- embed(diff(x), K) <=> _d_lag_matrix
    Z0 <- Z[, 1:P] #Z0 only picking the first # of stocks of cols, i.e the first difference data only
    ***/
    arma::mat Z1 = _d_lag_matrix.cols(P, _d_lag_matrix.n_cols - 1);
    arma::mat B = arma::ones<arma::mat>(Z1.n_rows, 1);
    Z1 = join_rows(B, Z1);
    // Z1 <- Z[, -c(1:P)] shed the first P cols
    // Z1 <- cbind(1, Z1) # Z1
    arma::mat ZK = _observation.rows(1, N - _lag);
    // ZK <- x[-N, ][K:(N - 1), ] # Zk

    _Z0 = Z0;
    _Z1 = Z1;
    _ZK = ZK;

    int n = Z0.n_rows;

    arma::mat M00 = Z0.t() * Z0 / n;
    arma::mat M11 = Z1.t() * Z1 / n;
    arma::mat Mkk = ZK.t() * ZK / n;
    arma::mat M01 = Z0.t() * Z1 / n;
    arma::mat M0k = Z0.t() * ZK / n;
    arma::mat Mk0 = ZK.t() * Z0 / n;
    arma::mat M10 = Z1.t() * Z0 / n;
    arma::mat M1k = Z1.t() * ZK / n;
    arma::mat Mk1 = ZK.t() * Z1 / n;
    arma::mat M11inv = arma::solve(M11, arma::eye<arma::mat>(size(M11)));

    arma::mat R0 = Z0 - (M01 * M11inv * Z1.t()).t();
    arma::mat Rk = ZK - (Mk1 * M11inv * Z1.t()).t();

    arma::mat S00 = M00 - M01 * M11inv * M10;
    arma::mat S0k = M0k - M01 * M11inv * M1k;
    arma::mat Sk0 = Mk0 - Mk1 * M11inv * M10;
    arma::mat Skk = Mkk - Mk1 * M11inv * M1k;

    Skk.raw_print(std::cout, "Skk:");
    Sk0.raw_print(std::cout, "Sk0:");
    S0k.raw_print(std::cout, "S0k:");
    S00.raw_print(std::cout, "S00:");

    arma::mat SkkInv = solve(Skk, arma::eye<arma::mat>(size(Skk)));
    arma::mat S00Inv = solve(S00, arma::eye<arma::mat>(S00.n_rows, S00.n_rows));

    _Pi= S0k * Skk.i();

    arma::uvec pivot; // could delete if not needed

    _C = pivoted_cholesky(Skk, 0.01, pivot);

    _eigenInput = _C.i() * (Sk0 * S00Inv * S0k) * _C.i().t();
    _eigenInput.raw_print(std::cout, "eigenInput");
}

// @TODO combine the getEigenOuput and getEigen Val
void VECM::getEigenOutput()
{
    eig_gen(_eigval, _eigvec, _eigenInput);
}

void VECM::getVorg()
{
    arma::mat real = arma::real(_eigvec);
    _Vorg = (solve(_C, eye(size(_C)))).t() * real;
}

arma::mat VECM::getStatistics()
{
    double N = _d_lag_matrix.n_rows;

    arma::mat stats;
    arma::mat eigen = arma::real(_eigval);

    arma::mat one = arma::ones<arma::mat>(size(eigen));
    arma::mat n = arma::mat(size(eigen));
    n.fill(-N);

    stats = n % log(one - eigen); // already negative

    return stats;
}


// refer to the structure of ca.jo@cvals, 1 = significant, 0 = insignficant
arma::mat VECM::getTest(arma::mat stats)
{
    int K = _observation.n_cols;

    arma::mat _test_stat = arma::mat(K, 3);
    for (int r = 0; r < K; r++){
        for (int c = 0; c < 3; c++){
            if (VECM::crit_eigen(r,c) < stats(r))
                _test_stat(r,c) = 1;
            else
                _test_stat(r,c) = 0;
        }
    }
    return _test_stat;
}
arma::vec VECM::getEigenValues()
{
    return arma::real(_eigval);
}

arma::mat VECM::getEigenVecMatrix()
{
    return arma::real(_eigvec);
}

arma::mat VECM::getVECModel()
{
    return _Gamma;
}

/*
int main()
{
    VECM vecm;
    vecm.loadCSV("GLD-GDX.csv"); // GLD-GDX.csv is the required file for testing the class
    vecm.compute(16);
}
*/

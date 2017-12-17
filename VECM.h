#include <armadillo>

// @TODO Should be using the OLS of the OLS class instead, sort out class dependency
arma::mat regressOLS(arma::mat X, arma::mat Y)
{
    // beta = (X.t() * X).i() * X.t() * Y;
    arma::mat beta;
    solve(beta, X.t() * X ,  X.t() * Y);

    return beta;
}

// @TODO This is not working at the moment
arma::mat regressGLS(arma::mat X, arma::mat Y, arma::mat covariance)
{
    arma::mat beta;

    arma::mat covarianceI;
    solve(covarianceI, covariance, eye(size(covariance)));

    solve(beta, X.t() * covarianceI * X, X.t() * covarianceI * Y);

    return beta;
}

// @TODO Ad hoc procedure, consider the class structure and hierarchy
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

// @TODO DOCUMENTATION of the process - The documentation of this class and the related VECM.cpp file
//       is deferred to a later stage, after all necessary details are implemented.

// For the current state of this class, please refer to the VEC.cpp file documentation

// This class is an implmentation of the URCA R package cajo.R file, with a plan to further implement
// FGLS - Feasible Generalized Least Square estimation instead of the current OLS method.

// Variable naming follows the cajo.R file in general.
class VECM {
public:
    VECM();
    VECM(arma::mat observation);
    ~VECM();

    arma::mat loadCSV(const std::string& filename);
    void compute(int nlags);

    arma::mat getTest(arma::mat stats);
    arma::vec getEigenValues();
    arma::mat getEigenVecMatrix();

    arma::mat getVECModel();

private:
    void saveMatCSV(arma::mat Mat, std::string filename);
    void saveMatCSV(arma::cx_mat Mat, std::string filename);

    //void preprocess();

    arma::mat computeCovarianceMatrix();
    arma::mat computeLagMatrix();
    arma::mat computeBeta();

    arma::mat computeVARPara();
    arma::mat computeGamma();
    arma::mat getMatrixDiff();
    arma::mat demean(arma::mat X);

    void getEigenInput();
    void getEigenOutput();
    void getVorg();

    arma::mat getStatistics();

private:
    unsigned int        _lag;
    arma::mat           _observation;

    arma::mat           _test_stat;
    arma::mat           _VARPara;
    arma::mat           _Gamma;
    arma::mat           _Pi;

    arma::mat           _C;
    arma::mat           _eigenInput;
    arma::cx_mat        _eigvec;
    arma::cx_vec        _eigval;
    arma::mat           _Vorg;

    arma::mat           _covariance;
    arma::mat           _beta;
    arma::mat           _lag_matrix;
    arma::mat           _d_lag_matrix;

    arma::mat           _Z0;
    arma::mat           _Z1;
    arma::mat           _ZK;

    static arma::mat    crit_eigen;
};

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

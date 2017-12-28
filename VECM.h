#include <armadillo>


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
    void saveMatCSV(arma::mat Mat, std::string filename);
    void saveMatCSV(arma::cx_mat Mat, std::string filename);
    
    void compute(int nlags);

    arma::mat getTest(arma::mat stats);
    arma::vec getEigenValues();
    arma::mat getEigenVecMatrix();

    arma::mat getVECModel();
    arma::mat getVorg();
    arma::mat getObservation();

private:
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



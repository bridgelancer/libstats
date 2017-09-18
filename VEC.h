#include <armadillo>

arma::mat regressOLS(const arma::mat&Y, );

class VEC {
public:
    VEC(const arma::mat& xMat);
    ~VEC();

    void doMaxEigenValueTest(int nlags);
    // Replace with do
    //void fitVECPara();
    //void fitVARPara();

    arma::vec getTestStat();
    arma::vec getEigenValue();
    arma::vec getEigenVecMatrix();

    arma::mat getVECModel();

private:
    void saveMatCSV(const arma::mat& Mat, std::string filename);
    void saveMatCSV(arma::cx_mat Mat, std::string filename);

    arma::mat regressGLS(const arma::mat& X, const arma::mat& Y, const arma::mat& covariance);

    arma::mat getCovarianceMatrix(const arma::mat& beta, const arma::mat& xMat, const arma::mat& lag);
    arma::mat getLagMatrix(const arma::mat& xMat, int nlags);
    arma::mat getBeta(const arma::mat& xMat, const arma::mat& lag, int nlags);

    arma::mat getVARPara(const arma::mat& xMat, const arma::mat& lag, const arma::mat& covariance);
    arma::mat getGamma(const arma::mat& VARPara);
    arma::mat getMatrixDiff(const arma::mat& xMat);

    arma::mat loadCSV(const std::string& filename);
    arma::mat demean(const arma::mat& X);

    arma::mat getEigenInput(const arma::mat& residualDX, const arma::mat& residualX);

    arma::cx_mat getEigenOutput(const arma::mat& eigenInput);
    arma::cx_vec getEigenVal(const arma::mat& eigenInput);

private:
    //static arma::mat crit_eigen;

    unsigned int    _lag;
    arma::mat       _test_stat;

    arma::mat       _VARPara;
    arma::mat       _VECPara;

    arma::cx_mat    _eigvec;
    arma::cx_vec    _eigval;

    arma::mat       _design;
    arma::mat       _observation;

    arma::mat       _beta;
    arma::mat       _lag_matrix;
    arma::mat       _d_lag_matrix;

    // still need to perform the statistical tests
};


// .cpp
VEC::crit_eigen = {{...}};


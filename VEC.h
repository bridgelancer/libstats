#include <armadillo>

class VEC
{
public:
    VEC(arma::mat xMat);
    ~VEC();

    void doMaxEigenValueTest(int nlags);
    void fitVECPara();
    void fitVARPara();
    
    arma::vec getOutStats();
    arma::vec getEigenValues();
    arma::vec getEigenVecMatrix();

    arma::mat getVECModel();

private:
    void saveMatCSV(arma::mat Mat, std::string filename);
    void saveMatCSV(arma::cx_mat Mat, std::string filename);

    arma::mat regressOLS(arma::mat X, arma::mat Y);
    arma::mat regressGLS(arma::mat X, arma::mat Y, arma::mat covariance);

    arma::mat getCovarianceMatrix(arma::mat beta, arma::mat xMat, arma::mat lag);
    arma::mat getLagMatrix(arma::mat xMat, int nlags);
    arma::mat getBeta(arma::mat xMat, arma::mat lag, int nlags);

    arma::mat getVARPara(arma::mat xMat, arma::mat lag, arma::mat covariance);
    arma::mat getVECMPara(arma::mat VARPara);
    arma::mat loadCSV(const std::string& filename);
    arma::mat getMatrixDiff(arma::mat xMat);
    arma::mat demean(arma::mat X);
    arma::mat getResidualDX(arma::mat xMat, arma::mat dLag, int nlags);
    arma::mat getResidualX(arma::mat xMat, arma::mat dLag, int nlags);
    arma::mat getEigenInput(arma::mat residualDX, arma::mat residualX);
    arma::cx_mat getEigenOutput(arma::mat eigenInput);
    arma::cx_vec getEigenVal(arma::mat eigenInput);

private:
    arma::mat VARPara;
    arma::mat VECPara;
    
    arma::cx_mat eigvec;
    arma::cx_vec eigval;

    // still need to perform the statistical tests
}
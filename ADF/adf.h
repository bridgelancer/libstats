//The class adf constructs the dickey-fuller and augmented dickey-fuller (ADF) tests as two distinct
//methods.

#include <armadillo>

#include "../OLS/ols.h"

class adf{
public:
    adf(std::string option);
    adf(std::string option, arma::vec y);
    adf(std::string option, arma::mat design_adf, arma::vec y);
    
    void getDesign_adf() const;
    void getObservation() const;
    void getBeta() const;
    
    void setDesign(arma::mat design);
    void setObservation(arma::vec observation);

    void evaluatePhi(std::string option);
    void evaluateSE(std::string option);
 
    void loadDesign(const std::string& filename);
    void saveBetaCSV();

    ols regression; //temporary public; for testing purposes only****
private:      
    arma::vec y; //The time-series data, notably the stock price
    double phi;
    double se_phi;
    
    std::string option;
    
    //ols regression; //call the ols(arma::mat design_adf, arma::vec y) constructor   

    arma::mat design_adf;
    arma::vec beta;    
};

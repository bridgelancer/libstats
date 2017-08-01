//The class ols constructs a object which process a design matrix of dimensions 
//m x n and observation vector m x 1 and evaulate the estimator vector beta of dimensions n x 1.

//This is currently a unstable version - further improvement should be osought on optimizing function
//loadDesign and the saveBetaCSV.

#include <armadillo>

class ols {
public:
    ols();
    
    void getDesign() const;
    void getObservation() const;
    void getBeta() const;
    
    void setDesign(arma::mat design);

    void setObservation(arma::vec observation);

    void evaluate();     
    
    void loadDesign(const std::string& filename); //parse the reference of the filename string to load function
    void saveBetaCSV() const;//save evaulated beta file to a csv of the same directory
            
private:
    arma::mat design;
    arma::vec observation;
    arma::vec beta;
};


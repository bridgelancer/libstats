#include "adf.h"

#include <armadillo>
#include <iostream>

adf::adf(std::string option):
    option(option),
    y(arma::vec()),
    design_adf(arma::mat()),
    beta(arma::vec()),
    regression(arma::mat(), arma::vec())
{ }
adf::adf(std::string option, arma::vec y):
    option(option),
    y(y),
    design_adf(arma::mat()),
    beta(arma::vec()),
    regression(arma::vec(), y)
{ }
adf::adf(std::string, arma::mat design_adf, arma::vec y):
    option(option),
    y(y),
    design_adf(design_adf),
    beta(arma::vec()),
    regression(design_adf, y)
{ }
void adf::getDesign_adf() const
{
    design_adf.print("design:");
}

void adf::getObservation() const
{ 
    y.print("observation:");
}

void adf::getBeta() const
{ 
    beta.print("beta:");
}

void adf::setDesign(arma::mat d)
{ 
    design_adf = d;
    regression.setDesign(d);
}

void adf::setObservation(arma::vec obs)
{ 
    y = obs;
    regression.setObservation(obs);
}

void adf::evaluatePhi(std::string option)
{ }

void adf::evaluateSE(std::string option)
{ }

void adf::loadDesign(const std::string& filename)
{ 
    arma::mat A = arma::mat();
    bool status = A.load(filename);

    if(status == true)
    {
        std::cout << "successfully loaded" << std::endl;
    }

    else
    {
        std::cout << "problem with loading" << std::endl;
    }
    
    design_adf = A;
    regression.setDesign(A);
}

void adf::saveBetaCSV()
{ 
    beta.save("beta.mat", arma::csv_ascii);
}

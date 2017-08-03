#include "ols.h"

#include <armadillo>
#include <iostream>

ols::ols():
    design(arma::mat()),
    observation(arma::vec()),
    beta(arma::vec())
{ }

ols::ols(arma::mat d, arma::vec obs):
    design(d),
    observation(obs),
    beta(arma::vec())
{ }

void ols::getDesign() const
{
    design.print("design:");
}

void ols::getObservation() const
{
    observation.print("observation:");
}

arma::vec ols::getBeta() const
{
    beta.print("beta:");
    return beta;
}

void ols::setDesign(arma::mat d)
{
    design = d;
}

void ols::setObservation(arma::vec obs)
{
    observation = obs;
}

void ols::evaluate()
{
    beta = ( design.t() * design ).i() * design.t() * observation; 
}

//need to be optimized, currently running at low speed
void ols::loadDesign(const std::string& filename)
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

    design = A;
}

//save beta to a csv file with filename "beta.mat"
void ols::saveBetaCSV() const
{
    beta.save("beta.mat", arma::csv_ascii);
}

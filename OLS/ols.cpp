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
    //beta = ( design.t() * design ).i() * design.t() * observation; 
    solve(beta, design.t() * design, design.t() * observation);
}

double ols::evaluateBetaSE()
{
    arma::vec error = arma::vec(observation.n_elem);

    error = observation - design * beta;

    double SEestimate;
    SEestimate = 1.0/(error.n_elem) * sum(square(error));
     
    double estimate;
    arma::mat store = SEestimate * (design.t() * design).i();

    estimate = sqrt(store(1,1)); //store is the covariance matrix that contains the SE values of all pair of estimators.
    return estimate;
 //assume independent and error term not skewed. the current calculated magnitude of staistics would be greater than intended.


/*
    double k;
    arma::mat bStore;
    
    bStore = beta * beta.t();  
    k = sum(square(error));
    std::cout << beta(beta.n_elem-1) << std::endl; 
    arma::mat tstore = (design.t() * design).i();
    arma::mat store = tstore * k * bStore * tstore;

    double estimate = sqrt(store(1,1));

    return estimate;
  */
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

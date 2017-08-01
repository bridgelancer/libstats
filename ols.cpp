#include "ols.h"

#include <armadillo>
#include <iostream>

ols::ols():
    design(arma::mat()),
    observation(arma::vec()),
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

void ols::getBeta() const
{
    beta.print("beta:");
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
    beta = (design.t()*design).i() * design.t() * observation; 
}


//not yet implemented
void ols::saveBetaCSV() const
{ }

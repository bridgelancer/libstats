#include "adf.h"

#include <armadillo>
#include <iostream>

adf::adf(OPTIONS option):
    option(option),
    y(arma::vec()),
    design_adf(arma::mat()),
    beta(arma::vec()),
    regression(arma::mat(), arma::vec()),
    k(0)
{ }

adf::adf(OPTIONS option, arma::vec y):
    option(option),
    y(y),
    design_adf(arma::mat()),
    beta(arma::vec()),
    regression(arma::vec(), y),
    k(0)
{ }

adf::adf(OPTIONS, arma::vec y, int k):
    option(option),
    y(y),
    design_adf(arma::mat()),
    beta(arma::vec()),
    regression(design_adf, y),
    k(k)
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

void adf::getStatistics() const
{
   std::cout << "The unit root t-test statistics is " << statistics << std::endl;
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

void adf::evaluateSE(int k)
{       
    int iter;
    int i = 0;
    arma::vec store = arma::vec(y.n_elem-4);
    for (iter = 4; iter< y.n_elem; ++iter){
        store(iter-4) = evaluatePhi(k, iter);
    } //not yet finished
    
    phi = store(store.n_elem-1);  //access final element
    std::cout << "The calculated value of phi is " << phi << std::endl;

    se_phi = stddev(store)/sqrt(y.n_elem);
    std::cout << "The standard error of phi is " << se_phi << std::endl;

    statistics = phi/se_phi;
 
}                

double adf::evaluatePhi(int k, int iter) 
{ 
    arma::vec x = y;
    x.resize(iter); //need to implement

    arma::vec y_ = arma::vec(x.n_elem); //y_ is the y_{t-1} of y
    
    y_(0) = 0;
    
    for (int i = 1; i<x.n_elem; i++){
        y_(i) = x(i-1);
    }
    
    switch (option) { 
        {case OPTIONS::DF:
            arma::vec product = x%y_;      
            
            arma::vec denom = x%x;
            phi = sum(product)/sum(denom);
                      
            return phi;
            break;
        }    
        
        {case OPTIONS::ADF:
            arma::mat fix = arma::mat(x.n_elem, 3); //n*3 arma matrix
            arma::mat lag = arma::mat(x.n_elem, k);; //n*k arma matrix, where k is the number of lag terms in consideration 
            
            for (int i = 0; i<x.n_elem; i++){
                fix(i,0) = 1;
                fix(i,1) = y_(i);
                fix(i,2) = i+1;
            } 
           //fix = [1,y_0,0; 1,y_1,1; ... ; 1,y_n-1,n]
            arma::vec y_plusone = arma::vec(x.n_elem+1);

            y_plusone(0) = 0;
            
            for (int i = 0; i < x.n_elem; i++){
                y_plusone(i+1) = x(i);
            }
                      
            for (int i = 0; i < x.n_elem; i++){              //loop through y.n_elem # of rows
                for(int count = 0; count <k; count++){     //loop through k # fo columns
                    if (i < count + 1)                           //if the time elapsed i is smaller than lagtime
                        lag(i,count) = 0;
                    else
                        lag(i,count) = y_plusone(i) - y_plusone(i-count-1);
                } 
            }
                
            lag.insert_cols(0, fix);           

            design_adf = lag;           
            

            regression.setDesign(design_adf);
            regression.setObservation(x);
            
            regression.evaluate();

            beta = regression.getBeta();
            
            return beta(1);            
            break;
        }
    }            
}

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

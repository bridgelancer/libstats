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

adf::adf(OPTIONS, arma::mat design_adf, arma::vec y, int k):
    option(option),
    y(y),
    design_adf(design_adf),
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

void adf::evaluateSE(OPTIONS option)
{ 
    switch (option) 
    { 
        {case OPTIONS::DF:
            ;
        }                
        {case OPTIONS::ADF:
            ;
        }
    }            
}

void adf::evaluatePhi(OPTIONS option, int k)
{ 
    std::cout << "evaluatePhi called"<< std::endl;

    arma::vec y_ = arma::vec(y.n_elem); //y_ is the y_{t-1} of y
    
    y_(0) = 0;
    
    for (int i = 1; i<y.n_elem; i++){
        y_(i) = y(i-1);
    }
    
    switch (option) { 
        {case OPTIONS::DF:
            arma::vec product = y%y_;       
            product.print();
            
            arma::vec denom = y%y;
            phi = sum(product)/sum(denom);
            std::cout << phi << std::endl;
            break;
        }    
        
        {case OPTIONS::ADF:
            arma::mat fix = arma::mat(y.n_elem, 3); //n*3 arma matrix
            arma::mat lag = arma::mat(y.n_elem, k);; //n*k arma matrix, where k is the number of lag terms in consideration 
            
            for (int i = 0; i<y.n_elem; i++){
                fix(i,0) = 1;
                fix(i,1) = y_(i);
                fix(i,2) = i+1;
            } 
            fix.print("fix:");

            //fix = [1,y_0,0; 1,y_1,1; ... ; 1,y_n-1,n]
            arma::vec y_plusone = arma::vec(y.n_elem+1);

            y_plusone(0) = 0;
            
            for (int i = 0; i < y.n_elem; i++){
                y_plusone(i+1) = y(i);
            }
            
            y_plusone.print("y_plusone");
            
            for (int i = 0; i < y.n_elem; i++){              //loop through y.n_elem # of rows
                for(int count = 0; count <k; count++){     //loop through k # fo columns
                    if (i < count + 1)                           //if the time elapsed i is smaller than lagtime
                        lag(i,count) = 0;
                    else
                        lag(i,count) = y_plusone(i) - y_plusone(i-count-1);
                } 
            }
            
            lag.print("lag:");
                
            lag.insert_cols(0, fix);

            lag.print("lag:");            

            design_adf = lag;           
            

            regression.setDesign(design_adf);
            regression.setObservation(y);
            
            regression.evaluate();

            beta = regression.getBeta();            
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

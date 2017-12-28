# Documentation for VECM class

The integration of the wrapper class **VECM** is currently underway and largely completed. The intended result of this integration would be a class that receives aggregated time series data from feed and determine whether the series is cointegrating.

This class should be constructed and setup **for testing cointegrating relationships**. The **VECM** class uses Johansesn Test to check whether cointegrating relationship exists for a particular portfolio of instruments.

The **VECM** class constructor consists of 1 input *_observation*. *_observation* is a arma::mat matrix that contains the time series of the portfolio. The initial (older) values should be put into the first rows of the *_observation* matrix.

The major computing function of the **VECM** class is the *compute* function. It takes an positive integer parameter *nlags* which should be determined from AIC/BIC or other suitable infomration criterion. Currently, no information criterion is implemented, and a pick of the lag term is done manually.

After the construction and computation at the back, the class stores its ouput to the various private members. The notable ones are *_Vorg* which should be output to the kalman filter wrapper class **kfWrapper**. The first column of *_Vorg* is the first eigenvector to be tested and so on. *test_stat* are the test results of the eigenvalue of tests of corresponding eigenvector. *_eigval* are the actual eigenvalues used to generate the test results of the three significance levels of *test_stat*. The actual coding of the Johansen Test refers to the R URCA package ca.jo.R method.

This class was warped from the file **VEC.cpp**. The validity of the output from **VEC.cpp** has been counterchecked against R implmentation. We used GLD_GDX.csv as the series, with nlags = 16. As mentioned before, the integration of this wrapper class is incomplete, and the checking of the relevant transformation from a matrix representation to one that suits the feed and system is still underway.

There are additional features that could be implmented to the class. Currently, the regression of the matrices in function *computeVARPara* uses Ordinary Least Square estimation (OLS). A Feasible Generalized Least Square (FGLS) method might be more accurate in estimating the real cointegrating relationship. The MATLAB's implementation of the Johansen Test uses FGLS. (There is also practical value in implementating GLS for other processes in the future.) 
#include "sigpack.h"
#include "VECM.h"
#include <fstream>
#include <iomanip>

using namespace arma;

// the first three functions would be removed from the real kalman class implmentation

// compare with 1 day before or 16 days before? Is it right to directly change hedge ratio using one day data?
// measurement error matrix = 0? other structure?
// validity of variable
// what is the real hedge ratio from VECM?

arma::mat loadCSV(const std::string& filename)
{ 
    arma::mat A = arma::mat();

    bool status = A.load(filename);

    if(status == true)
        std::cout << "Successfully loaded" << std::endl;
    else
        std::cout << "Problem with loading" << std::endl;

    return A;
}

void saveMatCSV(arma::mat Mat, std::string filename)
{ 
    std::ofstream stream = std::ofstream();
    stream.open(filename, std::ofstream::out | std::ofstream::trunc);

    int nrows = Mat.n_rows;
    int ncols = Mat.n_cols;

    stream << std::setprecision(4);
    stream.setf(std::ios::fixed, std::ios::floatfield);
    
    arma::vec maxVal = arma::vec(ncols);
    arma::vec minVal = arma::vec(ncols);
    arma::vec status = arma::vec(ncols);

    for (int j = 0; j < ncols; j++){
        maxVal(j) = max(arma::abs(Mat.col(j))); // finding the value of element in each column with largest magnitude
        minVal(j) = min(Mat.col(j)); // finding the smallest value

        if (maxVal(j) != std::abs(minVal(j))){ // if the value of largest magnitude is not the one of smallest value (i.e. +100/ -0.1)

            if ( maxVal(j) < 1 && std::abs(minVal(j)) < 1 && minVal(j) < 0.0)
                status(j) = 0;
            else if ( maxVal(j) < 1 && std::abs(minVal(j)) < 1 && minVal(j) > 0.0)
                status(j) = 1;
            else{
                double tmpMaxVal = maxVal(j);
                double tmpMinVal = std::abs(minVal(j));
                if ( maxVal(j) < 1 )
                     tmpMaxVal = maxVal(j) + 1;
                if ( std::abs(minVal(j)) < 1 )
                    tmpMinVal = std::abs(minVal(j)) + 1;

                if( (int)(log10(tmpMaxVal)) - (int)(log10(tmpMinVal)) < 1.0 && minVal(j) < 0.0)
                    status(j) = 0;
                else
                    status(j) = 1;
            }
        } 
        else // the value of largest magnitude is the one of smallest value (i.e. +10/ -100)
            status(j) = 0;
    }

    for (int i = 0; i < nrows; i++){
        for (int j = 0; j < ncols; j++){
            if (maxVal(j) > 1){
                if ( status(j) == 0 )
                    stream << std::setfill(' ') << std::setw( (int)log10(maxVal(j)) + (4 + 3) ) << Mat(i, j); // setting width, extra 3 spaces are added for storing ".", negative sign, and (int)log(x) rounds down;
                else if ( status(j) == 1 )
                    stream << std::setfill(' ') << std::setw( (int)log10(maxVal(j)) + (4 + 2) ) << Mat(i, j); // setting width, extra 2 spaces are added for storing ".", and (int)log(x) rounds down;
            }
            else {
                if ( status(j) == 0 )
                    stream << std::setfill(' ') << std::setw( (int)log10(1 + maxVal(j)) + (4 + 3) ) << Mat(i, j); // setting width, extra 3 spaces are added for storing ".", negative sign, and (int)log(x) rounds down;
                else if ( status(j) == 1 )
                    stream << std::setfill(' ') << std::setw( (int)log10(1 + maxVal(j)) + (4 + 2) ) << Mat(i, j); // setting width, extra 2 spaces are added for storing ".", and (int)log(x) rounds down;
            }

            if (j != ncols - 1)
                stream << ", ";
        }
        stream << "\n";
    }
    stream.close();
}


arma::mat getMatrixDiff(arma::mat lag, arma::mat xMat)
{
    int nrows = lag.n_rows;
    int ncols = lag.n_cols;

    arma::mat diff = arma::mat(nrows - 1, ncols);

    for (int i = 0; i < nrows - 1; i++){
        for (int j = 0; j < ncols; j++){
            diff(i , j) = lag(i+1 , j) - lag(i , j);
        }
    }

    return diff;
}   

arma::mat getLagMatrix(arma::mat xMat, int nlags)
{
    int nrows = xMat.n_rows;
    int ncols = xMat.n_cols;

    arma::mat lag = arma::mat(nrows - nlags + 1, ncols * nlags);

    int counter = 0;
    int counter2 = 0;
    // dunno whether deletes the last nlags row

    for (int i = 0; i < ncols * nlags; i++){
        for (int j = 0; j < nrows - nlags + 1; j++){
                lag(j, i) = xMat(j + counter, counter2);
        }
            counter2++;
            if ( (i+1) % ncols == 0){
                counter++;
                counter2 = 0;
            }
    }

    // add one rows of 1 behind lag matrix
    arma::mat B = arma::ones<arma::mat>(nrows - nlags + 1, 1);
    lag = join_rows(lag, B);
    
    return lag.submat(0,0, lag.n_rows-1 , 1);
}

using namespace std;
using namespace sp;
int main()
{
    VECM vecm;
    vecm.loadCSV("GLD-GDX.csv");
    vecm.compute(16);
    arma::mat hVec = vecm.getVorg(); //"hedge ratio" vector - @TODO check output format

    hVec.raw_print(std::cout, "hVec:");

    arma::uword N = 1;  // Nr of states
    arma::uword M = 1;  // Nr of measurements
    arma::uword L = 1;  // Nr of inputs

    // Instatiate a Kalman Filter
    KF kalman(N,M,L);

    // Initialisation and setup of system
    double P0 = 10; // multiplier of error covariance matrix
    double Q0 = 10; // multiplier of process noice matrix
    double R0 = 10; // multiplier of measurement noice matrix

    // Meas interval
    double dT = 1;

    arma::mat data = loadCSV("GLD-GDX.csv");
    arma::mat price_lagged = data.submat(16, 0 , 59, 1); //The 17th data to the 60th

    saveMatCSV(price_lagged, "price_lagged");

    // Number of samples
    // @TODO Sample should use differenced price data or raw price data?
    price_lagged.raw_print(std::cout, "price_lagged");
    arma::uword Nsamp = price_lagged.n_rows;

    std::cout << Nsamp << std::endl;

    arma::mat gamma = arma::zeros(price_lagged.n_rows, 1);
    gamma(0, 0) = hVec(0, 0)/hVec(1, 0);

    hVec.raw_print(std::cout, "hVec:");

    arma::mat init = {0.1166};
    kalman.set_state_vec(init.t());

    arma::mat A = {1};

    kalman.set_trans_mat(A); //set A as the state transition matrix (change of x/y based on hedge ratio)

    arma::mat H = price_lagged.col(1);

    // Determine covariance structure -> currently using one without covariance between variables (spec of VECM?)
    arma::mat P = P0*arma::eye(N,N); //set P = 10*In

    kalman.set_err_cov(P); //set error covariance matrix as P = 10*In

    arma::mat Q = arma::eye(N,N);
    Q = Q0*Q; //10 * diag[1,1]
    kalman.set_proc_noise(Q); //set process noice matrix as Q = some constant * identity

    arma::mat R = R0*arma::eye(1,1); // 25 * I -> R should not be zero
    kalman.set_meas_noise(R); //set measure noice matrix as R = 0

    arma::mat x_log(Nsamp - 1, N); //storing every state of x for each Nsamp
    arma::mat e_log(Nsamp - 1, 1); //storing every error associated with state x
    arma::cube P_log(N, N, Nsamp); //storing every error covariance matrix associated with state x
    arma::mat xs_log(Nsamp, M); //storing the smoothen x as xs
    arma::cube Ps_log(N, N, Nsamp);//storing the smoothen error covariance matrix as Ps

    // Kalman filter loop, with gamma changing in each update
    for(arma::uword n=0; n< Nsamp -1; n++)
    {
        kalman.get_state_vec().raw_print(std::cout, "x");
        kalman.set_meas_mat(H.row(n)); //set measurement matrix as H
        kalman.predict();
        //arma::mat measurement = arma::join_rows(arma::ones<mat>(1,1), price_lagged.submat(n+1, 0, n+1, 0));
        kalman.update(price_lagged.submat(n+1, 0, n+1, 0)); //price_lagged.row(n+1).t() is the "1/2" measurments obtained at time n+1 and fed to update
        // all state_vec, err, err_cov are updated
        x_log.row(n) = kalman.get_state_vec().t(); //updating x_log, the current state of x to be displayed
        e_log.row(n) = kalman.get_err().t(); //updating the error associated with state x
        P_log.slice(n) = kalman.get_err_cov(); //updating error_covariance matrices for each n

        kalman.get_state_vec().raw_print(std::cout, "x'");
    }
   
    // Consider whether we need any smoothing for Kalman fiter result to determine our strategy,
    // if smoothing is utilized, which type of smoothing is used
    saveMatCSV(x_log, "predict");
    saveMatCSV(e_log, "err");
    saveMatCSV(price_lagged, "measure");


    return 1;
}

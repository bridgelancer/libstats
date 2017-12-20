#include "sigpack.h"
#include "VECM.h"

using namespace arma;

arma::mat loadCSV(const std::string& filename)
{ 
    arma::mat A = arma::mat();

    bool status = A.load(filename);

    if(status == true)
    {
        std::cout << "Successfully loaded" << std::endl;
    }

    else
    {
        std::cout << "Problem with loading" << std::endl;
    }
    
    return A;
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
    arma::mat hVec = vecm.getEigenVecMatrix(); //"hedge ratio" vector

    arma::uword N = 2;  // Nr of states
    arma::uword M = 2;  // Nr of measurements
    arma::uword L = 1;  // Nr of inputs

    // Instatiate a Kalman Filter
    KF kalman(N,M,L);

    // Initialisation and setup of system
    double P0 = 100;
    double Q0 = 0.5;
    double R0 = 25;

    // Meas interval
    double dT = 1;

    // the data set should be [gamma, constant, delta X, delta Y]
    arma::mat data = loadCSV("GLD-GDX.csv");
    arma::mat price_lagged = getMatrixDiff(getLagMatrix(data, 16), data);

    // Number of samples
    price_lagged.raw_print(std::cout, "price_lagged");
    arma::uword Nsamp = price_lagged.n_rows;

    arma::mat gamma = arma::zeros(price_lagged.n_rows, 1);
    gamma(0, 0) = hVec(0)/hVec(1);

    kalman.set_state_vec(price_lagged.row(0).t()); //change

    arma::mat H =
    {
        {1, 0},
        {0, 1}
    };
    // Change arma::mat H to the measurmenet matrix relevant to our context (past price, VECM model?)
    kalman.set_meas_mat(H); //set measurement matrix as H

    // Determine covariance structure -> currently using one without covariance between variables (spec of VECM?)
    arma::mat P = P0*arma::eye(N,N); //set P = 100*In

    kalman.set_err_cov(P); //set error covariance matrix as P = 100*In

    arma::mat Q = arma::zeros(N,N);
    Q(0, 0) = 1;
    Q(1, 1) = 1;
    Q = Q0*Q; //0.5 * diag[1,1]
    kalman.set_proc_noise(Q); //set process noice matrix as Q = some constant * identity

    arma::mat R = R0*arma::zeros(M,M); // 25 * I
    kalman.set_meas_noise(R); //set measure noice matrix as R = 0

    arma::mat x_log(Nsamp, N); //storing every state of x for each Nsamp
    arma::mat e_log(Nsamp, M); //storing every error associated with state x
    arma::cube P_log(N, N, Nsamp); //storing every error covariance matrix associated with state x
    arma::mat xs_log(Nsamp, M); //storing the smoothen x as xs
    arma::cube Ps_log(N, N, Nsamp);//storing the smoothen error covariance matrix as Ps

    // Kalman filter loop
    for(arma::uword n=0; n< Nsamp; n++)
    {
        arma::mat A =
        {
            {1/gamma(n,0), 0    },
            {0           , gamma(n,0)},
        };
        kalman.set_trans_mat(A); //set A as the state transition matrix (we don't make prediction -> Identity)
        
        A.raw_print(std::cout, "A:");
        std::cout << gamma(n,0) << std::endl;

        kalman.predict();
        kalman.update(price_lagged.row(n).t()); //z.col(n) is the "2" measurments obtained at time n and feeded to update
        x_log.row(n) = kalman.get_state_vec().t(); //updating x_log, the current state of x to be displayed
        e_log.row(n) = kalman.get_err().t(); //updating the error associated with state x
        P_log.slice(n) = kalman.get_err_cov(); //updating error_covariance matrices for each n

        if (n < Nsamp -1)
            gamma.row(n+1) = x_log(n, 0) / x_log(n, 1); //check    
    }
   
    // Consider whether we need any smoothing for Kalman fiter result to determine our strategy,
    // If smoothing is utilized, which type of smoothing is used

    gamma.raw_print(std::cout, "gamma:");
    x_log.raw_print(std::cout, "x_log:");
    e_log.raw_print(std::cout, "e_log:");

    // RTS smoother
    kalman.rts_smooth(x_log,P_log,xs_log,Ps_log);

    return 1;
}

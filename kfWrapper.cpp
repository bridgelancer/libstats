#include "sigpack.h"
#include "VECM.h"
#include "kfWrapper.h"
#include <fstream>
#include <iomanip>

// This cpp file is intended for integration to the blackbox. The integration is still in progress

// The specification of this class kf refers to the P. Chan's Algorithmic Trading p.95

// measurement error matrix = 0? other structure?
// validity of variable

kfWrapper::kfWrapper(int N, int M, int L, arma::mat _observation, int _lag):
    N(N),
    M(M),
    L(L),
    kalman(sp::KF(N,M,L)),
    _lag(_lag),
    _x_log(),
    _e_log(),
    _P_log(),
    _observation(_observation)
{}

kfWrapper::~kfWrapper()
{}

// Vorg from VECM output; data from feed; _P0, _Q0, _R0 are tweakable parameters to tune the kalman filter
void kfWrapper::setKF(arma::mat Vorg, double _P0, double _Q0, double _R0)
{
    arma::mat hVec = Vorg; //"hedge ratio" vector - @TODO check output format

    // Initialisation and setup of system - (input parameters)
    double P0 = _P0; // multiplier of error covariance matrix 
    double Q0 = _Q0; // multiplier of process noice matrix
    double R0 = _R0; // multiplier of measurement noice matrix

    // Meas interval
    double dT = 1;

    // @TODO Hardcode for now, assuming there is a series
    // @TODO Change to receiving updates from datafeed

    double gamma = hVec(1, 0)/hVec(0, 0);

    arma::mat init = {0, -gamma};
    kalman.set_state_vec(init.t());

    arma::mat A = arma::eye (N, N);
    kalman.set_trans_mat(A); //set A as the state transition matrix (change of x/y based on hedge ratio)

    // Determine covariance structure -> currently using one without covariance between variables (spec of VECM?)
    arma::mat P = P0*arma::eye(N,N); //set P = 10*In
    kalman.set_err_cov(P); //set error covariance matrix as P = P0*In

    arma::mat Q = Q0* arma::eye(N,N);
    kalman.set_proc_noise(Q); //set process noice matrix as Q = some constant * identity

    arma::mat R = R0*arma::eye(1,1); // R = R0, should not be 0
    kalman.set_meas_noise(R); //set measurement noice "matrix" as R = 0

    //@TODO Consider factoirng out logging to avoid Nsamp dependency
    arma::mat x_log(1, N); //storing every state of x for each Nsamp
    arma::mat e_log(1, 1); //storing every error associated with state x
    arma::cube P_log(N, N, 1); //storing every error covariance matrix associated with state x

    _x_log = x_log;
    _e_log = e_log;
    _P_log = P_log;
}

//@TODO receive updated_price from feed, and update kalman filter accordingly
void kfWrapper::updateKF(arma::mat updated_price)
{   
    // Number of samples
    updated_price.raw_print(std::cout, "updated_price");
    
    // should be an arbitary number instead, or dynamic size matrix

    arma::mat H = arma::join_rows(arma::ones<arma::mat>(updated_price.n_rows, 1) , updated_price.col(1));
    
    // Kalman filter loop, with gamma changing in each update 
    
    // @TODO Factor out this - true should be replaced by suitable feed constraint and/or threading constraint

    kalmanLoop(updated_price);
      
    // Consider whether we need any smoothing for Kalman fiter result to determine our strategy,
    // if smoothing is utilized, which type of smoothing should be used   
}

void kfWrapper::kalmanLoop(arma::mat updated_price)
{
    this->kalman.set_meas_mat(arma::join_rows(arma::ones<arma::mat>(1, 1) , _observation)); //define H

    this->kalman.predict();
    this->kalman.update(updated_price);

    _x_log = arma::join_cols(_x_log, this->kalman.get_state_vec().t()); //updating x_log, the current state of x to be displayed
    _e_log = arma::join_cols(_e_log, this->kalman.get_err().t()); //updating the error associated with state x
    _P_log = arma::join_slices(_P_log, this->kalman.get_err_cov()); //updating error_covariance matrices for each n
    _observation = updated_price;
}


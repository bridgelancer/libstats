#include "sigpack.h"

using namespace std;
using namespace sp;

int main()
{
    // Number of samples - number of data points taken
    arma::uword Nsamp = 120;
    arma::uword N = 6;  // Nr of states determining the motion of series
    arma::uword M = 2;  // Nr of measurements that we measures (of states)
    arma::uword L = 1;  // Nr of inputs

    // Instatiate a Kalman Filter
    KF kalman(N,M,L);

    // Initialisation and setup of system
    double P0 = 100;
    double Q0 = 0.00003;
    double R0 = 25;

    // Meas interval
    double dT = 0.1;

    arma::mat x ={0, 10, 1, 50, -0.08, -9 };
    kalman.set_state_vec(0.9*x.t());
    // [X,Y,Vx,Vy,Ax,Ay] use position, velocity and accerleration as states
    // Change state vector arma::mat x to states relevant to our context (past price, VECM model parameters)
    arma::mat A =
    {
        {1, 0, dT,  0, dT*dT/2,       0},
        {0, 1,  0, dT,       0, dT*dT/2},
        {0, 0,  1,  0,      dT,       0},
        {0, 0,  0,  1,       0,      dT},
        {0, 0,  0,  0,       1,       0},
        {0, 0,  0,  0,       0,       1}
    };
    // Change arma::mat A to the transition matrix relevant to our context (VECm model design matrix?)
    kalman.set_trans_mat(A); //set A as the state transition matrix

    arma::mat H =
    {
        {1, 0, 0, 0, 0, 0 },
        {0, 1, 0, 0, 0, 0}
    };
    // Change arma::mat H to the measurmenet matrix relevant to our context (past price, VECM model?)
    kalman.set_meas_mat(H); //set measurement matrix as H

    // Determine covariance structure -> currently using one without covariance between variables (spec of VECM?)
    arma::mat P = P0*arma::eye(N,N); //set P = 100*In

    kalman.set_err_cov(P); //set error covariance matrix as P = 100*In

    arma::mat Q = arma::zeros(N,N);
    Q(N-2,N-2) = 1;
    Q(N-1,N-1) = 1;
    Q = Q0*Q; //0.00003 * diag[0,0,0,0,1,1]
    // Determine noice covariance structure

    // Might not be needed for initial setup, futher determine after successfully implemented Kalman filter
    kalman.set_proc_noise(Q); //set process noice matrix as Q

    arma::mat R = R0*arma::eye(M,M); // 25 * I
    kalman.set_meas_noise(R); //set measure noice matrix as R

        
    //This section and the associated variables should be replace by real measurment data for each Nsamp.
    //***********************************************************************************************************
    // Create simulation data
    // This part would be deleted
    arma::mat  z(M,Nsamp,arma::fill::zeros); // z of dimensions measurements * number of samples = 2 * 120
    arma::mat z0(M,Nsamp,arma::fill::zeros);
    //z, z0 are matrices set to be of dimensions (M, Nsamp) and filled with zeros
    
    arma::mat xx(N,1,arma::fill::zeros);
    
    xx = x.t();
    // This would be replaced by real data measurement
    for(arma::uword n=1; n< Nsamp; n++)
    {
       xx = A * xx+ 0.1 * Q * arma::randn(N,1); // for n, xx = A* XX + .1 * process noice matrix * random variable
       z0.col(n)    = H * xx;// column n of z0 = measurment matrix * xx
    }

    z.row(0) = z0.row(0)+ 0.001* R0 * arma::randn(1,Nsamp);  //1st row of z = 1st row of z0 + 0.001 * 25 * randnormal
    z.row(1) = z0.row(1)+ 0.8* R0 * arma::randn(1,Nsamp); //2nd row of z = 2nd row of z0 + 0.8 * 28 * randnormal

    //************************************************************************************************************
    
    arma::mat x_log(N,Nsamp); //storing every state of x for each Nsamp
    arma::mat e_log(M,Nsamp); //storing every error associated with state x
    arma::cube P_log(N,N,Nsamp); //storing every error covariance matrix associated with state x
    arma::mat xs_log(M,Nsamp); //storing the smoothen x as xs
    arma::cube Ps_log(N,N,Nsamp);//storing the smoothen error covariance matrix as Ps


    // Kalman filter loop
    for(arma::uword n=0; n< Nsamp; n++)
    {
        kalman.predict();
        kalman.update(z.col(n)); //z.col(n) is the "2" measurments obtained at time n and feeded to update

        x_log.col(n) = kalman.get_state_vec(); //updating x_log, the current state of x to be displayed
        e_log.col(n) = kalman.get_err(); //updating the error associated with state x
        P_log.slice(n) = kalman.get_err_cov(); //updating error_covariance matrices for each n
    }
   
    // Consider whether we need any smoothing for Kalman fiter result to determine our strategy,
    // If smoothing is utilized, which type of smoothing is used

    // RTS smoother
    kalman.rts_smooth(x_log,P_log,xs_log,Ps_log);
    
    return 1;
}

#include "sigpack.h"

using namespace std;
using namespace sp;

//IMPORTANT: remember to launch Xming and type "export DISPLAY=:0" into bash 

int main()
{
    // Number of samples
    arma::uword Nsamp = 120;
    arma::uword N = 6;  // Nr of states
    arma::uword M = 2;  // Nr of measurements
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
    arma::mat A =
    {
        {1, 0, dT,  0, dT*dT/2,       0},
        {0, 1,  0, dT,       0, dT*dT/2},
        {0, 0,  1,  0,      dT,       0},
        {0, 0,  0,  1,       0,      dT},
        {0, 0,  0,  0,       1,       0},
        {0, 0,  0,  0,       0,       1}
    };
    kalman.set_trans_mat(A); //set A as the state transition matrix

    arma::mat H = // only measure the positions
    {
        {1, 0, 0, 0, 0, 0 },
        {0, 1, 0, 0, 0, 0}
    };
    kalman.set_meas_mat(H); //set measurement matrix as H

    arma::mat P = P0*arma::eye(N,N); //set P = 100*In

    kalman.set_err_cov(P); //set error covariance matrix as P = 100*In

    arma::mat Q = arma::zeros(N,N);
    Q(N-2,N-2) = 1;
    Q(N-1,N-1) = 1;
    Q = Q0*Q; //0.00003 * diag[0,0,0,0,1,1]
    kalman.set_proc_noise(Q); //set process noice matrix as Q

    arma::mat R = R0*arma::eye(M,M); // 25 * I
    kalman.set_meas_noise(R); //set measure noice matrix as R

        
    //This *embraced section* and the associated variables should be replace by real measurment data for each Nsamp.
    //***********************************************************************************************************
    // Create simulation data
    arma::mat  z(M,Nsamp,arma::fill::zeros); // z of dimensions measurements * number of samples = 2 * 120
    arma::mat z0(M,Nsamp,arma::fill::zeros);
    //z, z0 are matrices set to be of dimensions (M, Nsamp) and filled with zeros

    arma::mat xx(N,1,arma::fill::zeros);
    xx = x.t();
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

    // RTS smoother
    kalman.rts_smooth(x_log,P_log,xs_log,Ps_log);

    // Display result
    gplot gp0;
    gp0.window("Plot", 10, 10, 500, 500);
    gp0.set_term("qt");
    gp0.plot_add(    z0.row(0),    z0.row(1),"True Y","lines dashtype 2 linecolor \"black\"");
    gp0.plot_add(     z.row(0),     z.row(1),"Meas Y","points");
    gp0.plot_add( x_log.row(0), x_log.row(1),"Kalman");
    gp0.plot_add(xs_log.row(0),xs_log.row(1),"RTS smooth");
    gp0.plot_show();

    return 1;
}

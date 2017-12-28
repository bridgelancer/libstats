#include <armadillo>

class kfWrapper {
public:
    kfWrapper(int N, int M, int L, arma::mat _observation, int _lag);
    ~kfWrapper();
    
    void setKF(arma::mat Vorg, double P0, double Q0, double R0);
    void updateKF(arma::mat updated_price);

    arma::mat get_x_log();
    arma::mat get_e_log();
    arma::mat get_P_log();

private:
    void kalmanLoop(arma::mat updated_price);

private:
    sp::KF              kalman;
    arma::uword         N;
    arma::uword         M;
    arma::uword         L;

    unsigned int        _lag;
    arma::mat           _data;

    arma::mat           _x_log;
    arma::mat           _e_log;
    arma::cube          _P_log;

    arma::mat           _observation;

};



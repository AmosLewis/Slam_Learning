#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;

//caculate
struct CURVE_FITTING_COST
{
    const double _x, _y;    // x,y data
    CURVE_FITTING_COST ( double x, double y ) : _x ( x ), _y ( y ) {}
    // calculate residual
    template <typename T>
    bool operator() (
            const T* const abc,     //  a b c coefficient parameter
            T* residual ) const     // residual
    {
        residual[0] = T ( _y ) - ceres::exp ( abc[0]*T ( _x ) *T ( _x ) + abc[1]*T ( _x ) + abc[2] ); // y-exp(ax^2+bx+c)
        return true;
    }

};

int main( int argc, char** argv)
{
    double a = 1.0, b = 2.0, c=1.0;     // real parameters
    int N = 100;                        // data point
    double w_sigma = 1.0;               // noise
    cv::RNG rng;                        // OpenCV random number
    double abc[3] = {0,0,0};            //  a b c estimation

    vector<double> x_data, y_data;      // data

    cout<<"genrating data:"<<endl;
    for( int i=0; i<N; i++ )
    {
        double x = i/100.0;
        x_data.push_back(x);
        y_data.push_back(
                exp( a*x*x + b*x + c) + rng.gaussian( w_sigma )
        );
        cout<<x_data[i]<<" "<<y_data[i]<<endl;
    }

    ceres::Problem problem;

    for( int i=0; i<N;i++ )
    {
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
                        new CURVE_FITTING_COST( x_data[i], y_data[i])
                ),
                nullptr,
                abc
        );
    }

    // set up Solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;       // how to solve Gauss Newton equation
    options.minimizer_progress_to_stdout = true;        // output to cout
    ceres::Solver::Summary summary;                     // output optimize information

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve ( options, &problem, &summary);        // start to
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

    chrono::duration<double> time_used = chrono::duration_cast<
            chrono::duration<double>>( t2-t1);
    cout<<"solve time cost = "<<time_used.count()<<"seconds."<<endl;

    // output result
    cout<<summary.BriefReport()<<endl;
    cout<<"estimated a,b,c = ";
    for (auto a:abc ) cout<<a<<" ";
    cout<<endl;

    return 0;
}
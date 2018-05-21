#ifndef TANSIGMOID_ACTIVATION_FUNCTION
#define TANSIGMOID_ACTIVATION_FUNCTION

#include "ActivationFunctionInterface.hpp"
#include <string>


class TanSigmoidActivationFunction: public ActivationFunctionInterface
{
protected:

public:
    TanSigmoidActivationFunction(){}
    ~TanSigmoidActivationFunction(){}

    std::string getIdCode(){return "tans";}

    double f(const double &in);

    double f1d(const double &in);

    double f2d(const double &in);

    double f3d(const double &in);

    void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1 = false, const bool flag_d2 = false, const bool flag_d3 = false);
};


#endif
#ifndef GAUSSIAN_ACTIVATION_FUNCTION
#define GAUSSIAN_ACTIVATION_FUNCTION

#include "ActivationFunctionInterface.hpp"
#include <string>


class GaussianActivationFunction: public ActivationFunctionInterface
{
protected:

public:
    GaussianActivationFunction(){}
    ~GaussianActivationFunction(){}

    std::string getIdCode(){return "gss";}

    // input should be in the rage [-3 : 3]
    double getIdealInputMu(){return 0.;}
    double getIdealInputSigma(){return 1.732050807568877;}

    // output is in the range [0 : 1] -> mu=0.5   sigma=1/(2*sqrt(3))
    double getOutputMu(){return 0.5;}
    double getOutputSigma(){return 0.288675134594813;}

    double f(const double &in);

    double f1d(const double &in);

    double f2d(const double &in);

    double f3d(const double &in);
};


#endif

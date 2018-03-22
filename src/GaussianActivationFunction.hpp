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
    double getIdealInputSigma(){return ;}

    double getOutputMu(){return ;}
    double getOutputSigma(){return ;}

    double f(const double &in);

    double f1d(const double &in);

    double f2d(const double &in);

    double f3d(const double &in);
};


#endif

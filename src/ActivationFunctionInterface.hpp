#ifndef ACTIVATION_FUNCTION_INTERFACE
#define ACTIVATION_FUNCTION_INTERFACE


#include <string>


class ActivationFunctionInterface
{
protected:

public:

    //return a 3-characters identification string
    virtual std::string getIdCode() = 0;

    // return the ideal input mean value (mu) and standard deviation (sigma)
    // (pretending a gaussian distribution)
    virtual double getIdealInputMu() = 0;
    virtual double getIdealInputSigma() = 0;

    // return the output mean value (mu) and standard deviation (sigma)
    // (pretending a gaussian distribution)
    virtual double getOutputMu() = 0;
    virtual double getOutputSigma() = 0;

    // compute the activation function value
    virtual double f(const double &) = 0;

    // first derivative of the activation function
    virtual double f1d(const double &) = 0;

    // second derivative of the activation function
    virtual double f2d(const double &) = 0;

    // third derivative of the activation function
    virtual double f3d(const double &) = 0;
};


#endif

#ifndef IDENTITY_ACTIVATION_FUNCTION
#define IDENTITY_ACTIVATION_FUNCTION


#include <string>


class IdentityActivationFunction: public ActivationFunctionInterface
{
protected:

public:
    IdentityActivationFunction(){}
    ~IdentityActivationFunction(){}

    std::string getIdCode(){return "id_";}

    // input can be assumed to be in the rage [-1 : 1] -> mu=0   sigma=1/sqrt(3)
    double getIdealInputMu(){return 0.;}
    double getIdealInputSigma(){return 0.577350269189626;}

    // output is identical to the input
    double getOutputMu(){return 0.;}
    double getOutputSigma(){return 0.577350269189626;}

    double f(const double &in){return in;}
    double f1d(const double &in){return 1.;}
    double f2d(const double &in){return 0.;}
    double f3d(const double &in){return 0.;}
};


#endif

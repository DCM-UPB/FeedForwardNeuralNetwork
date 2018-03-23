#ifndef OFFSET_UNIT_ACTIVATION_FUNCTION
#define OFFSET_UNIT_ACTIVATION_FUNCTION


#include <string>


class OffsetUnitActivationFunction: public ActivationFunctionInterface
{
protected:

public:
    OffsetUnitActivationFunction(){}
    ~OffsetUnitActivationFunction(){}

    std::string getIdCode(){return "off";}

    // input is not needed at all
    double getIdealInputMu(){return 0.;}
    double getIdealInputSigma(){return 0.;}

    // output mu and sigma are set to fulfil the formula requirements for the smart beta
    double getOutputMu(){return 1.;}
    double getOutputSigma(){return 1.;}

    double f(const double &in){return in;}
    double f1d(const double &in){return 1.;}
    double f2d(const double &in){return 0.;}
    double f3d(const double &in){return 0.;}
};


#endif

#ifndef NN_UNIT
#define NN_UNIT

#include "FedNetworkUnit.hpp"
#include "ActivationFunctionInterface.hpp"
#include "ActivationFunctionManager.hpp"
#include "NetworkUnitFeederInterface.hpp"

#include <stdexcept>
#include <string>

// Unit of an Artificial Neural Network
class NNUnit: public FedNetworkUnit
{
protected:

    // Activation Function of the unit
    // A function that calculates the output value from the input value (protovalue)
    ActivationFunctionInterface * _actf; // activation function

public:
    // Constructor and destructor
    NNUnit(ActivationFunctionInterface * actf = std_actf::provideActivationFunction(), NetworkUnitFeederInterface * feeder = NULL) : FedNetworkUnit(feeder) {if (actf) _actf = actf; else throw std::invalid_argument("NNUnit(): the parameter 'actf' was not valid");}
    NNUnit(const std::string &actf_id, NetworkUnitFeederInterface * feeder = NULL) : NNUnit(std_actf::provideActivationFunction(actf_id), feeder) {}
    virtual ~NNUnit(){ delete _actf; }

    // string code getters / setter
    virtual std::string getIdCode(){return "nnu";} // return identifier for unit type

    virtual std::string getMemberTreeIdCode(){return FedNetworkUnit::getMemberTreeIdCode() + " " + _actf->getTreeIdCode();} // append actf TreeIdCode
    virtual std::string getMemberTreeFullCode(){return FedNetworkUnit::getMemberTreeFullCode() + " " + _actf->getTreeFullCode();} // append actf TreeFullCode

    virtual void setMemberParams(const std::string &memberTreeFullCode);

    // Setters
    void setActivationFunction(ActivationFunctionInterface * actf){delete _actf; if (actf) _actf=actf; else throw std::invalid_argument("NNUnit::setActivationFunction(): the parameter 'actf' was not valid");}
    void setActivationFunction(const std::string &actf_id){this->setActivationFunction(std_actf::provideActivationFunction(actf_id));}

    // Getters
    ActivationFunctionInterface * getActivationFunction(){return _actf;}

    // Computation
    void computeOutput();
};


#endif

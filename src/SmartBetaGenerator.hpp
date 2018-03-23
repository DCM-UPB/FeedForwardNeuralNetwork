#ifndef SMART_BETA_GENERATOR
#define SMART_BETA_GENERATOR

#include "NNLayer.hpp"
#include "NNUnit.hpp"
#include "NNUnitFeederInterface.hpp"



namespace ffnn{

    // generate and set smart betas for the NNLayer L_target, which is connected to L_source
    void generateSmartBeta(NNLayer * L);

    // find the indexes of the units of a given NN layer which have a feeder
    std::vector<int> _findIndexesOfUnitsWithFeeder(NNLayer * L);

    // compute the optimal mu and sigma for a given unit's betas
    void _computeBetaMuAndSigma(NNUnit * U, double &mu, double &sigma);

    // set the beta of a feeder using a normal distribution with the given mean and standard deviation
    void _setRandomBeta(NNUnitFeederInterface * feeder, const double &mu, const double &sigma);

    // make the beta of feeder orthogonal to the beta of fixed_feeder
    void _makeBetaOrthogonal(NNUnitFeederInterface * fixed_feeder, NNUnitFeederInterface * feeder);

    // change the beta of feeder in order to get a mean and standard deviation as close as possible to mu and sigma
    // by only multiplying the beta vector for a scalar
    void _imposeMuAndSigma(NNUnitFeederInterface * feeder, const double &mu, const double &sigma);
}



#endif

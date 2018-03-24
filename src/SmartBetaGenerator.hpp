#ifndef SMART_BETA_GENERATOR
#define SMART_BETA_GENERATOR

#include "FeedForwardNeuralNetwork.hpp"
#include "NNLayer.hpp"
#include "NNUnit.hpp"
#include "NNUnitFeederInterface.hpp"



namespace smart_beta{

    // generate and set smart betas for the NNLayer L_target, which is connected to L_source
    void generateSmartBeta(FeedForwardNeuralNetwork * ffnn);
    void generateSmartBeta(NNLayer * L);



    // --- Internal methods that should not be called from outside

    // find the indexes of the units of a given NN layer which have a feeder
    std::vector<int> _findIndexesOfUnitsWithFeeder(NNLayer * L);

    // compute the optimal mu and sigma for a given unit's betas
    void _computeBetaMuAndSigma(NNUnit * U, double &mu, double &sigma);

    // set the beta of a feeder using a normal distribution with the given mean and standard deviation
    void _setRandomBeta(NNUnitFeederInterface * feeder, const double &mu, const double &sigma);

    // make the beta of feeder orthogonal to the beta of fixed_feeder, preserving the norm
    void _makeBetaOrthogonal(NNUnitFeederInterface * fixed_feeder, NNUnitFeederInterface * feeder);

}



#endif

#include <iostream>
#include <vector>
#include <assert.h>
#include <exception>
#include <cmath>
#include <numeric>

#include "FeedForwardNeuralNetwork.hpp"
#include "SmartBetaGenerator.hpp"
#include "ActivationFunctionManager.hpp"
#include "NNUnit.hpp"
#include "NNUnitFeederInterface.hpp"
#include "PrintUtilities.hpp"




int main(){
    using namespace std;

    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(3, 5, 3);
    ffnn->connectFFNN();
    ffnn->getLayer(1)->getUnit(4)->setActivationFunction(std_actf::provideActivationFunction("gss"));
    ffnn->getLayer(2)->getUnit(1)->setActivationFunction(std_actf::provideActivationFunction("gss"));



    // --- _findIndexesOfUnitsWithFeeder
    vector<int> idx;
    idx = smart_beta::_findIndexesOfUnitsWithFeeder(ffnn->getLayer(0));
    assert( idx.size() == 0);
    idx = smart_beta::_findIndexesOfUnitsWithFeeder(ffnn->getLayer(1));
    assert( idx.size() == 4);
    assert( idx[0] = 1 );
    assert( idx[0] = 2 );
    assert( idx[0] = 3 );
    assert( idx[0] = 4 );
    idx = smart_beta::_findIndexesOfUnitsWithFeeder(ffnn->getLayer(2));
    assert( idx.size() == 2);
    assert( idx[0] = 1 );
    assert( idx[0] = 2 );



    // --- _computeBetaMuAndSigma
    double mu, sigma;
    bool throw_exception;

    // input layer, without feeders
    for (int il=0; il<ffnn->getLayer(0)->getNUnits(); ++il){
        try {
            smart_beta::_computeBetaMuAndSigma(ffnn->getLayer(0)->getUnit(il), mu, sigma);
            throw_exception = false;
        } catch (exception &e) {
            throw_exception = true;
        }
        assert(throw_exception);
    }

    // hidden layer
    // offset unit
    try {
        smart_beta::_computeBetaMuAndSigma(ffnn->getLayer(1)->getUnit(0), mu, sigma);
        throw_exception = false;
    } catch (exception &e) {
        throw_exception = true;
    }
    assert(throw_exception);
    // 4 units
    for (int il=1; il<ffnn->getLayer(1)->getNUnits(); ++il){
        smart_beta::_computeBetaMuAndSigma(ffnn->getLayer(1)->getUnit(il), mu, sigma);
        double mu_denominator = 0.;
        for (int j=0; j<ffnn->getLayer(0)->getNUnits(); ++j){
            mu_denominator += ffnn->getLayer(0)->getUnit(j)->getActivationFunction()->getOutputMu();
        }
        const double mu_check = ffnn->getLayer(1)->getUnit(il)->getActivationFunction()->getIdealInputMu() / mu_denominator;
        double sigma_denominator = 0.;
        for (int j=0; j<ffnn->getLayer(0)->getNUnits(); ++j){
            sigma_denominator += ffnn->getLayer(0)->getUnit(j)->getActivationFunction()->getOutputSigma();
        }
        const double sigma_check = ffnn->getLayer(1)->getUnit(il)->getActivationFunction()->getIdealInputSigma() / sigma_denominator;
        assert( mu == mu_check );
        assert( sigma == sigma_check );
    }

    // output layer
    // offset unit
    try {
        smart_beta::_computeBetaMuAndSigma(ffnn->getLayer(2)->getUnit(0), mu, sigma);
        throw_exception = false;
    } catch (exception &e) {
        throw_exception = true;
    }
    assert(throw_exception);
    // 2 output units
    for (int il=1; il<ffnn->getLayer(2)->getNUnits(); ++il){
        smart_beta::_computeBetaMuAndSigma(ffnn->getLayer(2)->getUnit(il), mu, sigma);
        double mu_denominator = 0.;
        for (int j=0; j<ffnn->getLayer(1)->getNUnits(); ++j){
            mu_denominator += ffnn->getLayer(1)->getUnit(j)->getActivationFunction()->getOutputMu();
        }
        const double mu_check = ffnn->getLayer(2)->getUnit(il)->getActivationFunction()->getIdealInputMu() / mu_denominator;
        double sigma_denominator = 0.;
        for (int j=0; j<ffnn->getLayer(1)->getNUnits(); ++j){
            sigma_denominator += ffnn->getLayer(1)->getUnit(j)->getActivationFunction()->getOutputSigma();
        }
        const double sigma_check = ffnn->getLayer(2)->getUnit(il)->getActivationFunction()->getIdealInputSigma() / sigma_denominator;
        assert( mu == mu_check );
        assert( sigma == sigma_check );
    }



    // --- _setRandomBeta
    NNUnit * u = ffnn->getLayer(1)->getUnit(1);
    NNUnitFeederInterface * feeder = u->getFeeder();
    // compute mu and beta
    smart_beta::_computeBetaMuAndSigma(u, mu, sigma);
    // sample N times the beta
    const int N = 40000;
    vector<double> betas;
    for (int i=0; i<N; ++i){
        smart_beta::_setRandomBeta(feeder, mu, sigma);
        for (int ib=0; ib<feeder->getNBeta(); ++ib){
            betas.push_back(feeder->getBeta(ib));
        }
    }
    // compute the mean value of the betas and check that is equal to mu
    double mean = 0.;
    for (double b : betas){
        mean += b;
    }
    mean /= (N*feeder->getNBeta());
    // compute the standard deviation of the betas and check that is equal to mu
    double std_dev = 0.;
    for (double b : betas){
        std_dev += pow(b-mean, 2);
    }
    std_dev = sqrt(std_dev/(N*feeder->getNBeta()-1));
    // assertions
    assert( abs(mu-mean) < 0.05 );
    assert( abs(sigma-std_dev) < 0.01 );



    // --- _makeBetaOrthogonal
    // set random beta for a unit that will stay fixed
    NNUnit * fixed_u = ffnn->getLayer(1)->getUnit(2);
    NNUnitFeederInterface * fixed_feeder = fixed_u->getFeeder();
    smart_beta::_computeBetaMuAndSigma(fixed_u, mu, sigma);
    smart_beta::_setRandomBeta(fixed_feeder, mu, sigma);
    // set random beta for a unit that will be made orthogonal to the previously defined unit
    smart_beta::_computeBetaMuAndSigma(u, mu, sigma);
    smart_beta::_setRandomBeta(feeder, mu, sigma);
    // apply the orthogonaliy procedure
    smart_beta::_makeBetaOrthogonal(fixed_feeder, feeder);
    // compute the dot product and check that it is almost zero
    double dot_product = 0.;
    for (int i=0; i<feeder->getNBeta(); ++i){
        dot_product += feeder->getBeta(i) * fixed_feeder->getBeta(i);
    }
    assert( abs(dot_product) < 0.000000001);

    // now check that the norm is preserved by applying the orthogonality procedure

    // compute mu and beta
    smart_beta::_computeBetaMuAndSigma(u, mu, sigma);
    // sample N times the beta
    betas.clear();
    for (int i=0; i<N; ++i){
        smart_beta::_setRandomBeta(feeder, mu, sigma);
        smart_beta::_makeBetaOrthogonal(fixed_feeder, feeder);
        for (int ib=0; ib<feeder->getNBeta(); ++ib){
            betas.push_back(feeder->getBeta(ib));
        }
    }
    // compute the mean value of the betas and check that is equal to mu
    mean = 0.;
    for (double b : betas){
        mean += b;
    }
    mean /= (N*feeder->getNBeta());
    // compute the standard deviation of the betas and check that is equal to mu
    std_dev = 0.;
    for (double b : betas){
        std_dev += pow(b-mean, 2);
    }
    std_dev = sqrt(std_dev/(N*feeder->getNBeta()-1));
    // assertions
    assert( abs(mu-mean) < 0.05 );
    assert( abs(sigma-std_dev) < 0.01 );



    // --- generateSmartBeta
    // generate smart beta for the whole FFNN
    smart_beta::generateSmartBeta(ffnn);
    // check the orthogonality relations
    vector<double> betas2;
    for (int il=1; il<ffnn->getNLayers(); ++il){
        // cout << "layer " << il << endl;
        for (int iu1=1; iu1<ffnn->getLayer(il)->getNUnits()-1; ++iu1){
            betas.clear();
            for (int ib=0; ib<ffnn->getLayer(il)->getUnit(iu1)->getFeeder()->getNBeta(); ++ib)
                betas.push_back(ffnn->getLayer(il)->getUnit(iu1)->getFeeder()->getBeta(ib));
            for (int iu2=iu1+1; iu2<ffnn->getLayer(il)->getNUnits(); ++iu2){
                betas2.clear();
                for (int ib=0; ib<ffnn->getLayer(il)->getUnit(iu2)->getFeeder()->getNBeta(); ++ib)
                    betas2.push_back(ffnn->getLayer(il)->getUnit(iu2)->getFeeder()->getBeta(ib));
                const double dot_product = inner_product(begin(betas), end(betas), begin(betas2), 0.0);
                // cout << "    units " << iu1 << ", " << iu2 << "    ->    dot_product = " << dot_product << endl;
                // cout << "        expected " << (((iu1>ffnn->getLayer(il-1)->getNUnits()) || (iu2>ffnn->getLayer(il-1)->getNUnits())) ? true : false) << endl;
                // cout << endl;
                if ( !( (iu1>ffnn->getLayer(il-1)->getNUnits()) || (iu2>ffnn->getLayer(il-1)->getNUnits()) ) ){
                    assert( dot_product < 0.000000001 );
                }
            }
        }
    }



    delete ffnn;

    return 0;
}

#include <iostream>
#include <assert.h>
#include <cmath>

#include "FeedForwardNeuralNetwork.hpp"
#include "ActivationFunctionManager.hpp"
#include "PrintUtilities.hpp"



int main(){
    using namespace std;

    // make a check while the FFNN is not connected yet
    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(3, 5, 3);
    ffnn->pushHiddenLayer(4);
    ffnn->getLayer(0)->getUnit(2)->setActivationFunction(std_actf::provideActivationFunction("lgs"));
    ffnn->getLayer(1)->getUnit(3)->setActivationFunction(std_actf::provideActivationFunction("gss"));
    ffnn->getLayer(2)->getUnit(1)->setActivationFunction(std_actf::provideActivationFunction("gss"));
    ffnn->getLayer(3)->getUnit(2)->setActivationFunction(std_actf::provideActivationFunction("gss"));

    FeedForwardNeuralNetwork * ffnn2 = new FeedForwardNeuralNetwork(ffnn);

    assert(ffnn->getNLayers() == ffnn2->getNLayers());
    assert(ffnn->getLayer(0)->getNUnits() == ffnn2->getLayer(0)->getNUnits());
    assert(ffnn->getLayer(1)->getNUnits() == ffnn2->getLayer(1)->getNUnits());
    assert(ffnn->getLayer(2)->getNUnits() == ffnn2->getLayer(2)->getNUnits());
    assert(ffnn->getLayer(3)->getNUnits() == ffnn2->getLayer(3)->getNUnits());

    assert(!ffnn2->isConnected());
    assert(!ffnn2->hasFirstDerivativeSubstrate());
    assert(!ffnn2->hasSecondDerivativeSubstrate());
    assert(!ffnn2->hasVariationalFirstDerivativeSubstrate());

    assert(ffnn->getLayer(0)->getUnit(0)->getActivationFunction()->getIdCode() == "off");
    assert(ffnn->getLayer(0)->getUnit(1)->getActivationFunction()->getIdCode() == "id_");
    assert(ffnn->getLayer(0)->getUnit(2)->getActivationFunction()->getIdCode() == "lgs");
    assert(ffnn->getLayer(1)->getUnit(0)->getActivationFunction()->getIdCode() == "off");
    assert(ffnn->getLayer(1)->getUnit(1)->getActivationFunction()->getIdCode() == "lgs");
    assert(ffnn->getLayer(1)->getUnit(2)->getActivationFunction()->getIdCode() == "lgs");
    assert(ffnn->getLayer(1)->getUnit(3)->getActivationFunction()->getIdCode() == "gss");
    assert(ffnn->getLayer(1)->getUnit(4)->getActivationFunction()->getIdCode() == "lgs");
    assert(ffnn->getLayer(2)->getUnit(0)->getActivationFunction()->getIdCode() == "off");
    assert(ffnn->getLayer(2)->getUnit(1)->getActivationFunction()->getIdCode() == "gss");
    assert(ffnn->getLayer(2)->getUnit(2)->getActivationFunction()->getIdCode() == "lgs");
    assert(ffnn->getLayer(2)->getUnit(3)->getActivationFunction()->getIdCode() == "lgs");
    assert(ffnn->getLayer(3)->getUnit(0)->getActivationFunction()->getIdCode() == "off");
    assert(ffnn->getLayer(3)->getUnit(1)->getActivationFunction()->getIdCode() == "lgs");
    assert(ffnn->getLayer(3)->getUnit(2)->getActivationFunction()->getIdCode() == "gss");

    delete ffnn2;




    // connect the FFNN and make a check
    ffnn->connectFFNN();

    ffnn2 = new FeedForwardNeuralNetwork(ffnn);

    assert(ffnn2->isConnected());
    assert(!ffnn2->hasFirstDerivativeSubstrate());
    assert(!ffnn2->hasSecondDerivativeSubstrate());
    assert(!ffnn2->hasVariationalFirstDerivativeSubstrate());

    assert(ffnn->getNBeta() == ffnn2->getNBeta());
    for (int i=0; i<ffnn2->getNBeta(); ++i){
        assert(ffnn->getBeta(i) == ffnn2->getBeta(i));
    }

    delete ffnn2;




    // add first derivative substrate and make a check
    ffnn->addFirstDerivativeSubstrate();

    ffnn2 = new FeedForwardNeuralNetwork(ffnn);

    assert(ffnn2->isConnected());
    assert(ffnn2->hasFirstDerivativeSubstrate());
    assert(!ffnn2->hasSecondDerivativeSubstrate());
    assert(!ffnn2->hasVariationalFirstDerivativeSubstrate());

    delete ffnn2;




    // add variational first derivative substrate and make a check
    ffnn->addVariationalFirstDerivativeSubstrate();

    ffnn2 = new FeedForwardNeuralNetwork(ffnn);

    assert(ffnn2->isConnected());
    assert(ffnn2->hasFirstDerivativeSubstrate());
    assert(!ffnn2->hasSecondDerivativeSubstrate());
    assert(ffnn2->hasVariationalFirstDerivativeSubstrate());

    delete ffnn2;




    // add second derivative substrate and make a check
    ffnn->addSecondDerivativeSubstrate();

    ffnn2 = new FeedForwardNeuralNetwork(ffnn);

    assert(ffnn2->isConnected());
    assert(ffnn2->hasFirstDerivativeSubstrate());
    assert(ffnn2->hasSecondDerivativeSubstrate());
    assert(ffnn2->hasVariationalFirstDerivativeSubstrate());




    // check that, once the input is the same for the two FFNN, all outputs (including derivatives) are the same
    const double input[2] = {37.42, -18.01};
    ffnn->setInput(input);
    ffnn->FFPropagate();
    ffnn2->setInput(input);
    ffnn2->FFPropagate();

    assert(ffnn->getOutput(0) == ffnn2->getOutput(0));
    assert(ffnn->getOutput(1) == ffnn2->getOutput(1));

    assert(ffnn->getFirstDerivative(0, 0) == ffnn2->getFirstDerivative(0, 0));
    assert(ffnn->getFirstDerivative(0, 1) == ffnn2->getFirstDerivative(0, 1));
    assert(ffnn->getFirstDerivative(1, 0) == ffnn2->getFirstDerivative(1, 0));
    assert(ffnn->getFirstDerivative(1, 1) == ffnn2->getFirstDerivative(1, 1));

    assert(ffnn->getSecondDerivative(0, 0) == ffnn2->getSecondDerivative(0, 0));
    assert(ffnn->getSecondDerivative(0, 1) == ffnn2->getSecondDerivative(0, 1));
    assert(ffnn->getSecondDerivative(1, 0) == ffnn2->getSecondDerivative(1, 0));
    assert(ffnn->getSecondDerivative(1, 1) == ffnn2->getSecondDerivative(1, 1));

    for (int i=0; i<ffnn->getNBeta(); ++i){
        assert(ffnn->getVariationalFirstDerivative(0, i) == ffnn2->getVariationalFirstDerivative(0, i));
        assert(ffnn->getVariationalFirstDerivative(1, i) == ffnn2->getVariationalFirstDerivative(1, i));
    }


    delete ffnn2;



    delete ffnn;

    return 0;
}

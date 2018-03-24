#include <iostream>
#include <cmath>
#include <fstream>

#include "FeedForwardNeuralNetwork.hpp"
#include "PrintUtilities.hpp"



class MyActivationFunction: public ActivationFunctionInterface
{
protected:

public:

    std::string getIdCode(){
        return "mya";
    }

    // range [-2 : 2]
    double getIdealInputMu(){return 0.;};
    double getIdealInputSigma(){return 1.154700538379252;};

    // range [0 : 4]
    double getOutputMu(){return 2.;};
    double getOutputSigma(){return 1.154700538379252;};

    double f(const double &in){
        // function
        return 2.*in*in;
    }

    double f1d(const double &in){
        // first derivative
        return 4.*in;
    }

    double f2d(const double &in){
        // second derivative
        return 4.;
    }

    double f3d(const double &in){
        // third derivative
        return 0.;
    }
};



int main() {
    using namespace std;



    cout << "Let's start by creating a Feed Forward Artificial Neural Netowrk (FFANN)" << endl;
    cout << "========================================================================" << endl;
    cin.ignore();

    cout << "We generate a FFANN with 4 layers and 3, 4, 5, 2 units respectively" << endl;
    cin.ignore();

    // NON I/O CODE
    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(3, 4, 2);
    ffnn->pushHiddenLayer(5);
    //

    cout << "Graphically it looks like this" << endl;
    cin.ignore();
    printFFNNStructure(ffnn);
    cout << endl << endl;
    cin.ignore();



    cout << "Access an activation function" << endl;
    cout << "=============================" << endl;
    cin.ignore();

    cout << "We want to access the activation function of the 2nd unit of the 3rd layer.";
    cin.ignore();

    ActivationFunctionInterface * actf = ffnn->getLayer(2)->getUnit(1)->getActivationFunction();
    cout << "The activation function we accessed has id = " << actf->getIdCode() << endl;
    cin.ignore();

    cout << "Now we want to substitue it with our activation function (MyActivationFunction)";
    cin.ignore();

    MyActivationFunction * myactf = new MyActivationFunction();
    ffnn->getLayer(2)->getUnit(1)->setActivationFunction(myactf);
    cout << "Done!";
    cin.ignore();

    cout << "Graphically the NN looks like this" << endl;
    cin.ignore();
    printFFNNStructure(ffnn);
    cout << endl << endl;

    cout << endl << endl;


    return 0;
}

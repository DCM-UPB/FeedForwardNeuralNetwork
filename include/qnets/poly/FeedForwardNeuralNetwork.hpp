#ifndef FFNN_NET_FEEDFORWARDNEURALNETWORK_HPP
#define FFNN_NET_FEEDFORWARDNEURALNETWORK_HPP

#include "qnets/poly/actf/ActivationFunctionInterface.hpp"
#include "qnets/poly/fmap/FeatureMapLayer.hpp"
#include "qnets/poly/layer/FedLayer.hpp"
#include "qnets/poly/layer/InputLayer.hpp"
#include "qnets/poly/layer/NNLayer.hpp"
#include "qnets/poly/layer/NetworkLayer.hpp"
#include "qnets/poly/layer/OutputNNLayer.hpp"
#include "qnets/poly/unit/NetworkUnit.hpp"

#include <cstddef>
#include <string>
#include <vector>

class FeedForwardNeuralNetwork
{
private:
    void _construct(const int &insize, const int &hidlaysize, const int &outsize); // construct from minimal set of unit numbers
    void _registerLayer(NetworkLayer * newLayer, const int &indexFromBack = 0); // register layers to correct vectors, position controlled by indexFromBack
    void _addNewLayer(const std::string &idCode, const int &nunits, const int &indexFromBack = 0, const std::string &params = ""); // creates and registers a new layer according to idCode and nunits
    void _addNewLayer(const std::string &idCode, const std::string &params = "", const int &indexFromBack = 0); // creates and registers a new layer according to idCode and params code (without it the layer will only have an offset unit)
    void _updateNVP(); // internal method to update _nvp member, call it after you changed/created variational parameter assignment
protected:
    std::vector<NetworkLayer *> _L; // contains all kinds of layers
    std::vector<FedLayer *> _L_fed; // contains layers with feeder
    std::vector<NNLayer *> _L_nn; // contains neural layers
    std::vector<FeatureMapLayer *> _L_fm; // contains feature map layers
    InputLayer * _L_in = nullptr; // input layer
    OutputNNLayer * _L_out = nullptr; // output layer

    bool _flag_connected = false;  // flag that tells if the FFNN has been connected or not
    bool _flag_1d = false, _flag_2d = false, _flag_v1d = false, _flag_c1d = false, _flag_c2d = false;  // flag that indicates if the substrates for the derivatives have been activated or not

    int _nvp = 0;  // global number of variational parameters

public:
    FeedForwardNeuralNetwork(const int &insize, const int &hidlaysize, const int &outsize);
    explicit FeedForwardNeuralNetwork(const char * filename);  // file must be formatted as with the method storeOnFile()
    explicit FeedForwardNeuralNetwork(const FeedForwardNeuralNetwork &ffnn);
    explicit FeedForwardNeuralNetwork(const FeedForwardNeuralNetwork * ffnn): FeedForwardNeuralNetwork(*ffnn) {}

    ~FeedForwardNeuralNetwork();


    // --- Get information about the NN structure
    int getNLayers() const { return _L.size(); }
    int getNFedLayers() const { return _L_fed.size(); }
    int getNNeuralLayers() const { return _L_nn.size(); }
    int getNFeatureMapLayers() const { return _L_fm.size(); }
    int getNHiddenLayers() const { return _L_nn.size() - 1; }

    int getNInput() const { return _L_in->getNInputUnits(); }
    int getNOutput() const { return _L_out->getNOutputNNUnits(); }
    int getLayerSize(const int &li) const { return _L[li]->getNUnits(); }

    NetworkLayer * getLayer(const int &li) { return _L[li]; }
    FedLayer * getFedLayer(const int &li) { return _L_fed[li]; }
    NNLayer * getNNLayer(const int &li) { return _L_nn[li]; }
    FeatureMapLayer * getFeatureMapLayer(const int &li) { return _L_fm[li]; }
    InputLayer * getInputLayer() { return _L_in; }
    OutputNNLayer * getOutputLayer() { return _L_out; }

    bool isConnected() const { return _flag_connected; }
    bool hasFirstDerivativeSubstrate() const { return _flag_1d; }
    bool hasSecondDerivativeSubstrate() const { return _flag_2d; }
    bool hasVariationalFirstDerivativeSubstrate() const { return _flag_v1d; }
    bool hasCrossFirstDerivativeSubstrate() const { return _flag_c1d; }
    bool hasCrossSecondDerivativeSubstrate() const { return _flag_c2d; }


    // --- Modify NN structure
    void setGlobalActivationFunctions(ActivationFunctionInterface * actf);
    void pushHiddenLayer(const int &size);
    void popHiddenLayer();
    void pushFeatureMapLayer(const int &size, const std::string &params = "");


    // --- Connect the neural network
    void connectFFNN();
    void disconnectFFNN();


    // --- Manage the betas, which exist only after that the FFNN has been connected
    int getNBeta() const;
    double getBeta(const int &ib) const;
    void getBeta(double * beta) const;
    void setBeta(const int &ib, const double &beta);
    void setBeta(const double * beta);
    void randomizeBetas(); // has to be changed maybe if we add beta that are not "normal" weights

    // --- Manage the variational parameters (which may contain a subset of beta and/or non-beta parameters),
    //     which exist only after that they are assigned to actual parameters in the network (e.g. betas)
    void assignVariationalParameters(const int &starting_layer_index = 0); // make betas variational parameters, starting from starting_layer
    int getNVariationalParameters() const { return _nvp; }
    double getVariationalParameter(const int &ivp) const;
    void getVariationalParameter(double * vp) const;
    void setVariationalParameter(const int &ivp, const double &vp);
    void setVariationalParameter(const double * vp);


    // --- Substrates for the calculations of derivatives
    void addFirstDerivativeSubstrate();  // coordinates first derivatives
    void addSecondDerivativeSubstrate();  // coordinates second derivatives

    // Substrate for the variational derivative d/dbeta:
    void addVariationalFirstDerivativeSubstrate();  // variational first derivatives

    // Substrate for the cross derivatives d/dx d/dbeta
    void addCrossFirstDerivativeSubstrate();  // cross first derivatives
    void addCrossSecondDerivativeSubstrate();  // cross second derivatives

    // shortcut for (connecting and) adding substrates
    void addSubstrates(bool flag_d1 = false, bool flag_d2 = false, bool flag_vd1 = false, bool flag_c1d = false, bool flag_c2d = false);
    void connectAndAddSubstrates(bool flag_d1 = false, bool flag_d2 = false, bool flag_vd1 = false, bool flag_c1d = false, bool flag_c2d = false);


    // Set initial parameters
    void setInput(const double * in);
    void setInput(const int &i, const double &in);

    // --- Computation
    void FFPropagate();

    // Shortcut for computation: set input and get all values and derivatives with one calculations.
    // If some derivatives are not supported (substrate missing) the values will be leaved unchanged.
    void evaluate(const double * in, double * out = nullptr, double ** d1 = nullptr, double ** d2 = nullptr, double ** vd1 = nullptr);


    // --- Get outputs
    void getOutput(double * out) const;
    double getOutput(const int &i) const;

    void getFirstDerivative(double ** d1) const;
    void getFirstDerivative(const int &iu, double * d1) const;  // iu is the unit index
    double getFirstDerivative(const int &iu, const int &i1d) const; // i is the index of the output elemnet (i.e. unit=1, offset unit is meaningless), i1d the index of the input element

    void getSecondDerivative(double ** d2) const;
    void getSecondDerivative(const int &i, double * d2) const;  // i is the output index
    double getSecondDerivative(const int &i, const int &i2d) const; // i is the index of the output element, i2d the index of the input element

    void getVariationalFirstDerivative(double ** vd1) const;
    void getVariationalFirstDerivative(const int &i, double * vd1) const;  // i is the output index
    double getVariationalFirstDerivative(const int &i, const int &iv1d) const;  // i is the index of the output element, iv1d the index of the beta element

    void getCrossFirstDerivative(double *** d1vd1) const;
    void getCrossFirstDerivative(const int &i, double ** d1vd1) const;  // i is the output index
    void getCrossFirstDerivative(const int &i, const int &i1d, double * d1vd1) const;  // i is the output index, i1d the input index
    double getCrossFirstDerivative(const int &i, const int &i1d, const int &iv1d) const;  // i is the index of the output element, i1d, of the input element, iv1d the index of the beta element

    void getCrossSecondDerivative(double *** d2vd1) const;
    void getCrossSecondDerivative(const int &i, double ** d2vd1) const;  // i is the output index
    void getCrossSecondDerivative(const int &i, const int &i2d, double * d2vd1) const;  // i is the output index, i2d the input index
    double getCrossSecondDerivative(const int &i, const int &i2d, const int &iv1d) const;  // i is the index of the output element, i2d, of the input element, iv1d the index of the beta element


    // --- Store FFNN on file
    void storeOnFile(const char * filename, bool store_betas = true) const;
};


#endif

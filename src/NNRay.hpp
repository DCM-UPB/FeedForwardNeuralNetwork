#ifndef NN_RAY
#define NN_RAY

#include "NNUnit.hpp"
#include "NNLayer.hpp"
#include "NNUnitFeederInterface.hpp"

#include <vector>
#include <random>

class NNRay: public NNUnitFeederInterface{
protected:
    // random number generator, used to initialize the intensities
    std::random_device _rdev;
    std::mt19937_64 _rgen;
    std::uniform_real_distribution<double> _rd;

    // key component of the ray: the source and their intensisities
    std::vector<NNUnit *> _source;   // units from which the ray takes the values from
    std::vector<double> _intensity;   // intensity of each sorgent unit, i.e. its weight
    std::vector<int> _intensity_id;  // intensity identification id, useful for the NN
    int _intensity_id_shift;  // shift of the previous vector

public:
    NNRay(NNLayer * nnl);
    virtual ~NNRay();

    // sources
    int getNSources(){return _source.size();}
    NNUnit * getSource(const int &i){return _source[i];}

    // beta
    int getNBeta(){return _intensity.size();}
    double getBeta(const int &i){return _intensity[i];}
    void setBeta(const int &i, const double &b){_intensity[i]=b;}

    // Variational Parameters
    int getNVariationalParameters(){return _intensity.size();}
    int setVariationalParametersIndexes(const int &starting_index);
    bool getVariationalParameterValue(const int &id, double &value);
    bool setVariationalParameterValue(const int &id, const double &value);

    // Computation
    double getFeed();
    double getFirstDerivativeFeed(const int &i1d);
    double getSecondDerivativeFeed(const int &i2d);
    double getVariationalFirstDerivativeFeed(const int &iv1d);
    double getCrossFirstDerivativeFeed(const int &i1d, const int &iv1d);
    double getCrossSecondDerivativeFeed(const int &i2d, const int &iv2d);

    // Beta Index
    bool isBetaIndexUsedInThisRay(const int &id);   // beta is directly used?
    bool isBetaIndexUsedForThisRay(const int &id);   // beta is used directly or indirectly?

};


#endif

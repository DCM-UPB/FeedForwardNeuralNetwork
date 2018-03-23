#include "NNLayer.hpp"

#include "NNRay.hpp"
#include "ActivationFunctionManager.hpp"

#include <iostream>


// --- Variational Parameters

bool NNLayer::setVariationalParameter(const int &id, const double &vp)
{
    std::vector<NNUnit *>::size_type i=1;
    NNUnitFeederInterface * feeder;
    bool flag = false;
    while ( (!flag) && (i<_U.size()) )
        {
            feeder = _U[i]->getFeeder();
            if (feeder)
                {
                    flag = feeder->setVariationalParameterValue(id,vp);
                }
            i++;
        }
    return flag;
}


bool NNLayer::getVariationalParameter(const int &id, double &vp)
{
    std::vector<NNUnit *>::size_type i=1;
    NNUnitFeederInterface * feeder;
    bool flag = false;
    while ( (!flag) && (i<_U.size()) )
        {
            feeder = _U[i]->getFeeder();
            if (feeder)
                {
                    flag = feeder->getVariationalParameterValue(id, vp);
                }
            i++;
        }
    return flag;
}


int NNLayer::getNVariationalParameters()
{
    int nvp=0;
    NNUnitFeederInterface * feeder;
    for (std::vector<NNUnit *>::size_type i=1; i<_U.size(); ++i)
        {
            feeder = _U[i]->getFeeder();
            if (feeder)
                {
                    nvp += feeder->getNVariationalParameters();
                }
        }
    return nvp;
}


// --- Computation

void NNLayer::computeValues()
{
    for (NNUnit * u : _U) u->computeValues();
}


// --- Values to compute

int NNLayer::setVariationalParametersID(const int &id_vp)
{
    int id = id_vp;
    for (std::vector<NNUnit *>::size_type i=1; i<_U.size(); ++i)
        {
            id = _U[i]->getFeeder()->setVariationalParametersIndexes(id);
        }
    return id;
}


void NNLayer::addCrossSecondDerivativeSubstrate(const int &nx0, const int &nvp)
{
    for (std::vector<NNUnit *>::size_type i=0; i<_U.size(); ++i)
        {
            _U[i]->setCrossSecondDerivativeSubstrate(nx0, nvp);
        }
}


void NNLayer::addCrossFirstDerivativeSubstrate(const int &nx0, const int &nvp)
{
    for (std::vector<NNUnit *>::size_type i=0; i<_U.size(); ++i)
        {
            _U[i]->setCrossFirstDerivativeSubstrate(nx0, nvp);
        }
}


void NNLayer::addVariationalFirstDerivativeSubstrate(const int &nvp)
{
    for (std::vector<NNUnit *>::size_type i=0; i<_U.size(); ++i)
        {
            _U[i]->setVariationalFirstDerivativeSubstrate(nvp);
        }
}


void NNLayer::addSecondDerivativeSubstrate(const int &nx0)
{
    for (std::vector<NNUnit *>::size_type i=0; i<_U.size(); ++i)
        {
            _U[i]->setSecondDerivativeSubstrate(nx0);
        }
}


void NNLayer::addFirstDerivativeSubstrate(const int &nx0)
{
    for (std::vector<NNUnit *>::size_type i=0; i<_U.size(); ++i)
        {
            _U[i]->setFirstDerivativeSubstrate(nx0);
        }
}


// --- Connection

void NNLayer::connectOnTopOfLayer(NNLayer * nnl)
{
    NNUnitFeederInterface * ray;
    for (std::vector<NNUnit *>::size_type i=1; i<_U.size(); ++i)
        {
            ray = new NNRay(nnl);
            _U[i]->setFeeder(ray);
        }
}


void NNLayer::disconnect()
{
    NNUnitFeederInterface * ray;
    for (std::vector<NNUnit *>::size_type i=1; i<_U.size(); ++i)
        {
            ray = _U[i]->getFeeder();
            delete ray;
            _U[i]->setFeeder(NULL);
        }
}


// --- Modify structure

void NNLayer::setActivationFunction(ActivationFunctionInterface * actf)
{
    _U[0]->setActivationFunction(&std_actf::id_actf);
    for (std::vector<NNUnit *>::size_type i=1; i<_U.size(); ++i)
        {
            _U[i]->setActivationFunction(actf);
        }
}


void NNLayer::setSize(const int &nunits)
{
    ActivationFunctionInterface * actf = _U[1]->getActivationFunction();
    for (std::vector<NNUnit *>::size_type i=0; i<_U.size(); ++i)
        {
            delete _U[i];
        }
    _U.clear();
    _U.push_back(new NNUnit(&std_actf::id_actf));
    _U[0]->setProtoValue(1.);
    for (int i=1; i<nunits; ++i)
        {
            _U.push_back(new NNUnit(actf));
        }
}


// --- Getters



// --- Constructor

NNLayer::NNLayer(const int &nunits, ActivationFunctionInterface * actf)
{
    _U.push_back(new NNUnit(&std_actf::offset_actf));
    _U[0]->setProtoValue(1.);

    for (int i=1; i<nunits; ++i)
        {
            _U.push_back(new NNUnit(actf));
        }
}


// --- Destructor

NNLayer::~NNLayer()
{
    for (std::vector<NNUnit *>::size_type i=0; i<_U.size(); ++i)
        {
            delete _U[i];
        }
    _U.clear();
}

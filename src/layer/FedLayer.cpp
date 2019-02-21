#include "ffnn/layer/FedLayer.hpp"
#include "ffnn/feed/FeederInterface.hpp"
#include "ffnn/unit/FedUnit.hpp"

#include <vector>


// --- Register Unit

void FedLayer::_registerUnit(NetworkUnit * newUnit)
{
    NetworkLayer::_registerUnit(newUnit);
    if(FedUnit * fu = dynamic_cast<FedUnit *>(newUnit)) {
        _U_fed.push_back(fu);
    }
}


// --- Variational Parameters

bool FedLayer::setVariationalParameter(const int &id, const double &vp)
{
    std::vector<FedUnit *>::size_type i=0;
    bool flag = false;
    while ( (!flag) && (i<_U_fed.size()) )
        {
            FeederInterface * feeder = _U_fed[i]->getFeeder();
            if (feeder) {
                flag = feeder->setVariationalParameterValue(id,vp);
            }
            i++;
        }
    return flag;
}


bool FedLayer::getVariationalParameter(const int &id, double &vp)
{
    std::vector<FedUnit *>::size_type i=0;
    bool flag = false;
    while ( (!flag) && (i<_U_fed.size()) )
        {
            FeederInterface * feeder = _U_fed[i]->getFeeder();
            if (feeder) {
                flag = feeder->getVariationalParameterValue(id, vp);
            }
            i++;
        }
    return flag;
}


int FedLayer::getNVariationalParameters()
{
    int nvp=0;
    for (std::vector<FedUnit *>::size_type i=0; i<_U_fed.size(); ++i)
        {
            FeederInterface * feeder = _U_fed[i]->getFeeder();
            if (feeder) {
                nvp += feeder->getNVariationalParameters();
            }
        }
    return nvp;
}

int FedLayer::getMaxVariationalParameterIndex()
{
    int max_index = -1;
    for (std::vector<FedUnit *>::size_type i=0; i<_U_fed.size(); ++i)
        {
            FeederInterface * feeder = _U_fed[i]->getFeeder();
            if (feeder) {
                int index = feeder->getMaxVariationalParameterIndex();
                if (index > max_index) max_index = index;
            }
        }
    return max_index;
}


// --- Values to compute

int FedLayer::setVariationalParametersID(const int &id_vp)
{
    int id = id_vp;
    for (std::vector<FedUnit *>::size_type i=0; i<_U_fed.size(); ++i)
        {
            FeederInterface * feeder = _U_fed[i]->getFeeder();
            if (feeder) {
                id = _U_fed[i]->getFeeder()->setVariationalParametersIndexes(id);
            }
        }
    return id;
}


// --- Connection

void FedLayer::connectOnTopOfLayer(NetworkLayer * nl)
{
    for (std::vector<FedUnit *>::size_type i=0; i<_U_fed.size(); ++i)
        {
            FeederInterface * feeder = this->connectUnitOnTopOfLayer(nl, i); // note that i==0 means the first non-offset unit
            if (feeder) _U_fed[i]->setFeeder(feeder);
        }
}

void FedLayer::disconnect()
{
    for (std::vector<FedUnit *>::size_type i=0; i<_U_fed.size(); ++i)
        {
            delete _U_fed[i]->getFeeder();
            _U_fed[i]->setFeeder(NULL);
        }
}

#ifndef IDENTITY_MAP_UNIT
#define IDENTITY_MAP_UNIT

#include "ffnn/fmap/FeatureMapUnit.hpp"
#include "ffnn/fmap/IdentityMap.hpp"

#include <string>

class IdentityMapUnit: public FeatureMapUnit<IdentityMap>
{
public:
    // string code id
    std::string getIdCode(){return "IDMU";} // return identifier for unit type
};

#endif
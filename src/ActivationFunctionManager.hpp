#ifndef ACTIVATION_FUNCTION_MANAGER
#define ACTIVATION_FUNCTION_MANAGER


#include "ActivationFunctionInterface.hpp"
#include "IdentityActivationFunction.hpp"
#include "LogisticActivationFunction.hpp"
#include "GaussianActivationFunction.hpp"
#include "OffsetUnitActivationFunction.hpp"

#include <string>


namespace std_actf{

    extern OffsetUnitActivationFunction offset_actf;
    extern IdentityActivationFunction id_actf;
    extern LogisticActivationFunction lgs_actf;
    extern GaussianActivationFunction gss_actf;

    ActivationFunctionInterface * provideActivationFunction(const std::string idcode);
}



#endif

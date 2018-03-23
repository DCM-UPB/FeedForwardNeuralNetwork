#include "ActivationFunctionManager.hpp"




namespace std_actf{


    OffsetUnitActivationFunction offset_actf = OffsetUnitActivationFunction();
    IdentityActivationFunction id_actf = IdentityActivationFunction();
    LogisticActivationFunction lgs_actf = LogisticActivationFunction();
    GaussianActivationFunction gss_actf = GaussianActivationFunction();


    ActivationFunctionInterface * provideActivationFunction(const std::string idcode){
        if (idcode == offset_actf.getIdCode()){
            return &offset_actf;
        }

        if (idcode == id_actf.getIdCode()){
            return &id_actf;
        }

        if (idcode == lgs_actf.getIdCode()){
            return &lgs_actf;
        }

        if (idcode == gss_actf.getIdCode()){
            return &gss_actf;
        }

        return 0;
    }

}

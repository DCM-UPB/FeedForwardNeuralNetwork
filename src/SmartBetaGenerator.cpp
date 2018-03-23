#include "SmartBetaGenerator.hpp"

#include "NNUnitFeederInterface.hpp"
#include "ActivationFunctionInterface.hpp"

#include <random>
#include <numeric>
#include <cmath>


namespace ffnn{

    void generateSmartBeta(NNLayer * L){
        using namespace std;

        // get the indexes of the units we are interested in
        vector<int> idx = _findIndexesOfUnitsWithFeeder(L);

        if (idx.size() < 1){
            // TODO throw exception
        }

        // set the first unit beta
        NNUnit * u0 = L->getUnit(idx[0]);
        double mu, sigma;
        _computeBetaMuAndSigma(u0, mu, sigma);
        _setRandomBeta(u0->getFeeder(), mu, sigma);

        // set all the other unit beta
        const int n_li_units = ( signed(idx.size()) <= u0->getFeeder()->getNBeta() ) ? signed(idx.size()) : u0->getFeeder()->getNBeta();
        for (int i=1; i<n_li_units ; ++i){
            NNUnit * u = L->getUnit(idx[i]);
            double mu, sigma;
            _computeBetaMuAndSigma(u, mu, sigma);
            _setRandomBeta(u->getFeeder(), mu, sigma);
            for (int j=0; j<i; ++j){
                _makeBetaOrthogonal(L->getUnit(idx[j])->getFeeder(), u->getFeeder());
            }
            _imposeMuAndSigma(u->getFeeder(), mu, sigma);
        }
    }


    std::vector<int> _findIndexesOfUnitsWithFeeder(NNLayer * L){
        /*
        Given a NN layer, find the indexes of the units which have a feeder
        */
        using namespace std;

        vector<int> idx;
        for (int i=0; i<L->getNUnits(); ++i){
            if (L->getUnit(i)->getFeeder() != 0){
                idx.push_back(i);
            }
        }

        return idx;
    }


    void _computeBetaMuAndSigma(NNUnit * U, double &mu, double &sigma){
        /*
        Use the formulas:
            mu_beta = mu_actf_input / (sum_U_sources mu_actf_output)
            sigma_beta = sigma_actf_input / (sum_U_sources mu_actf_output)
        */
        NNUnitFeederInterface * feeder = U->getFeeder();

        // first compute the denominators
        double mu_denominator = 0;
        double sigma_denominator = 0;
        ActivationFunctionInterface * actf;
        for (int j=0; j<feeder->getNSources(); ++j){
            actf = feeder->getSource(j)->getActivationFunction();
            mu_denominator += actf->getOutputMu();
            sigma_denominator += actf->getOutputSigma();
        }

        // assign the correct values
        mu = U->getActivationFunction()->getIdealInputMu() / mu_denominator;
        sigma = U->getActivationFunction()->getIdealInputSigma() / sigma_denominator;
    }


    void _setRandomBeta(NNUnitFeederInterface * feeder, const double &mu, const double &sigma){
        /*
        Sample the beta of a given feeder using a normal distribution with mean mu and standard deviation sigma.
        */
        using namespace std;

        default_random_engine gen;
        normal_distribution<double> norm(mu, sigma);

        for (int i=0; i<feeder->getNBeta(); ++i){
            feeder->setBeta(i, norm(gen));
        }
    }


    void _makeBetaOrthogonal(NNUnitFeederInterface * fixed_feeder, NNUnitFeederInterface * feeder){
        /*
        Modify feeder in order to make it orthogonal to fixed_feeder
        */
        using namespace std;

        // build vectors v and u, corresponding to the betas of the two feeders
        vector<double> v;
        for (int i=0; i<fixed_feeder->getNBeta(); ++i){
            v.push_back(fixed_feeder->getBeta(i));
        }
        vector<double> u;
        for (int i=0; i<feeder->getNBeta(); ++i){
            u.push_back(feeder->getBeta(i));
        }

        // make u orthogonal to v
        const double u_v = inner_product(begin(u), end(u), begin(v), 0.0);
        const double v_v = inner_product(begin(v), end(v), begin(v), 0.0);
        const double orthog_fact = u_v/v_v;
        for (unsigned int i=0; i<u.size(); ++i){
            u[i] -= orthog_fact * v[i];
        }

        // set the values of u to the feeder
        for (int i=0; i<feeder->getNBeta(); ++i){
            feeder->setBeta(i, u[i]);
        }
    }

    void _imposeMuAndSigma(NNUnitFeederInterface * feeder, const double &mu, const double &sigma){
        /*
        Change the beta of feeder in order to get as close as possible to a mean mu and a standard deviation sigma.
        This is done without changing the direction of the vector beta, which means that it is obtained by multiplying
        beta for a scalar lambda, requiring that the cost function
            (mu - mean)^2 + (sigma - standard deviation)^2
        is minimised. This corresponds to
            lambda = (mu * mean + sigma * standard deviation) / (mean^2 + standard deviation^2)
        */
        using namespace std;

        // build vectors u corresponding to the betas of the feeder
        vector<double> u;
        for (int i=0; i<feeder->getNBeta(); ++i){
            u.push_back(feeder->getBeta(i));
        }

        // compute the mean
        double mean = 0;
        for (double b : u){
            mean += b;
        }
        mean /= u.size();

        // compute the stadard deviation
        double std_dev = 0;
        for (double b : u){
            std_dev += pow(b - mean, 2);
        }
        std_dev = sqrt( std_dev / (u.size() - 1.) );

        // compute lambda
        const double lambda = ( mu*mean + sigma*std_dev ) / ( pow(mean, 2) + pow(std_dev, 2) ) ;

        // multiply beta by the scalar lambda
        for (int i=0; i<feeder->getNBeta(); ++i){
            feeder->setBeta(i, lambda*u[i]);
        }
    }

}

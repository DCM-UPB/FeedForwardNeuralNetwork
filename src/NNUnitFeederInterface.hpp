#ifndef NN_UNIT_FEEDER_INTERFACE
#define NN_UNIT_FEEDER_INTERFACE


class NNUnitFeederInterface
{
protected:

public:
    virtual ~NNUnitFeederInterface(){}

    // get info on beta
    virtual int getNBeta() = 0;
    virtual double getBeta(const int &i) = 0;
    virtual void setBeta(const int &i, const double &b) = 0;

    // feed
    virtual double getFeed() = 0;  // get   sum_j( b_j x_j )

    // derivatives
    virtual double getFirstDerivativeFeed(const int &i) = 0;  // get   sum_j( b_j dx_j/dx0_i ), where x0 is the coordinate in the input layer
    virtual double getSecondDerivativeFeed(const int &i) = 0;  // get   sum_j( b_j d^2x_j/dx0_j^2 ), where x0 is the coordinate in the input layer

    // variational parameters
    virtual int getNVariationalParameters() = 0;  // return the number of variational parameters involved
    virtual int setVariationalParametersIndexes(const int &starting_index) = 0;  // set the identification index of each variational parameter starting from the given input value
    // return the index that the next feeder might take as input
    virtual bool getVariationalParameterValue(const int &id, double &value) = 0; // get the variational parameter with identification index id and store it in value
    // return true if the parameters has been found, false otherwise
    virtual bool setVariationalParameterValue(const int &id, const double &value) = 0; // set the variational parameter with identification index id with the number stored in value
    // return true if the parameters has been found, false otherwise
    // derivative in respect to the variational parameters
    virtual double getVariationalFirstDerivativeFeed(const int &i) = 0;  // get   sum_j( db_j/db_i x_j + b_j dx_j/db_i ), where i is the index of the variational parameter (j might not cross it)
};


#endif

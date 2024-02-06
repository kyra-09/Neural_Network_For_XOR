#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include "layer.h"
#include <functional>

class Activation : public Layer
{
public:
    Activation () = default;
    Activation (std::function<double(double)> activation , std::function<double(double)> activation_prime)
    {
          this -> activation = activation;
          this -> activation_prime = activation_prime;
    }

    virtual ~Activation() = default;
    virtual Vector forward (const Vector& input) override ;
    virtual Vector backward ( Vector& output_gradient
                            , double learning_rate) override ;

private:
 Vector activation_input ;
std::function <double(double)> activation;
std::function <double(double)> activation_prime;
};

#endif
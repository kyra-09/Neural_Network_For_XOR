#ifndef ACTIVATION_FUNTIONS_H 
#define ACTIVATION_FUNCTIONS_H

#include "activation_layer.h"
#include "layer.h"

class Tanh : public Activation 
{
public:
   Tanh() : Activation (
      [](double x) {return std::tanh(x);},
      [](double x) {double tanhx = std::tanh(x);
                    return 1.0 - tanhx * tanhx ;}
   ) {}

   virtual ~Tanh() = default;
};

#endif
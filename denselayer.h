
#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "layer.h"
#include <iostream>

class Dense_layer : public Layer
{
public:
    Dense_layer() = default;
    Dense_layer( size_t input_size , size_t output_size) ;
    virtual ~Dense_layer() = default;

    virtual Vector forward (const Vector& input) override ;
    virtual Vector backward ( Vector& output_gradient
                            , double learning_rate) override ;


protected:
    Matrix weights ;
    Vector bias;
    Vector self_input;
    size_t input_size ;
    size_t output_size ;

    double Generate_Random_Normal ();
    Vector Generate_Random_Vector ();
    Matrix Generate_Random_Matrix ();
    Matrix Transpose (const Matrix& transpose);
};

#endif
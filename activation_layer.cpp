#include "activation_layer.h"

Vector Activation::forward (const Vector& input)
{
    activation_input = input;

   Vector result (input.size() , 0.00);
    for (size_t i = 0 ; i < input.size() ; ++i)
    {
        result[i] = activation(input[i]);
    }

    return result;
}

Vector Activation::backward ( Vector& output_gradient , double learning_rate)
{
    Vector input_grad (output_gradient.size() , 0.00);
    for(size_t i = 0; i < output_gradient.size() ; ++i){
        input_grad[i] = output_gradient[i] * activation_prime(activation_input[i]);
    }
       
    return input_grad;
}
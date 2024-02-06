#include "denselayer.h"
#include <iostream>

Dense_layer::Dense_layer(size_t input_size , size_t output_size)
           :input_size(input_size) , output_size(output_size) 
{
    weights = Generate_Random_Matrix();
    bias =  Generate_Random_Vector();
}


std::vector<std::vector<double>> Dense_layer::Transpose(const std::vector<std::vector<double>>& matrix) {
    std::vector<std::vector<double>> result(matrix[0].size(), std::vector<double>(matrix.size()));
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}



Vector Dense_layer::forward (const Vector& input) 
{

    self_input = input ;

   Vector result (output_size , 0.00);
    for (size_t i = 0 ; i < output_size ; ++i) {
        for (size_t j = 0 ; j < input_size ; ++j){
            result[i] += weights[i][j] * input[i];
        }
        result[i] += bias[i];
    }

    return result;
}

Vector  Dense_layer::backward (  Vector& output_gradient
                                     , double learning_rate) 
{
    Matrix weight_grad (output_size , Vector(input_size , 0.00));
    Vector input_grad (input_size , 0.00);


    // Computing the gradient of the weights
    for (size_t i = 0; i < weight_grad.size(); ++i) {
    for (size_t j = 0; j < weight_grad[0].size(); ++j) {
        weight_grad[i][j] += output_gradient[i] * self_input[j]; // Element-wise multiplication
    }
   }


    //computing input gradient.
    Matrix transpose_weights = Transpose(weights);

    for (size_t i = 0 ; i < transpose_weights.size() ; ++i){
        for(size_t j= 0 ; j < transpose_weights[0].size() ; ++j){
            input_grad[i] += transpose_weights[i][j] * output_gradient[j];
        }
    }


    //updating weights - 2d matrix
    for (size_t i = 0 ; i < weights.size() ; ++i){
        for (size_t j = 0 ; j < weights[0].size() ; ++j){
            weights[i][j] -= learning_rate * weight_grad[i][j];
        }
    }

    
    //updating bias - 1D matrix
    for (size_t i = 0 ; i < bias.size() ; ++i){
        bias[i] -= learning_rate * output_gradient[i];
    }

    return input_grad;

}

double Dense_layer::Generate_Random_Normal()
{
   static std::random_device rd;
   static std::mt19937 gen(rd());
   std::normal_distribution <double> distribution (0.0 , 1.0);
   return distribution(gen);
}


Vector Dense_layer::Generate_Random_Vector ()
{
   Vector temp (output_size , 0.00);
    for (size_t i = 0 ; i < output_size ; ++i)
    {
        temp[i] = Generate_Random_Normal ();
    }
    return temp;
}


Matrix Dense_layer::Generate_Random_Matrix ()
{
    Matrix temp (output_size , Vector(input_size, 0.00));
    for (size_t i = 0 ; i < output_size ; ++i) {
      for (size_t j = 0 ; j < input_size ; ++j) {
        temp[i][j] = Generate_Random_Normal ();
      }
    }
    return temp;
}


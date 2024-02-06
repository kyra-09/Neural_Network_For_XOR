#include "denselayer.h"
#include "activation_layer.h"
#include "activation_functions.h"
#include "layer.h"
#include "mse.h"

#include <iostream>

int main()
{
    Matrix X =      {{0.0 , 0.0},
                     {0.0 , 1.0},
                     {1.0 , 0.0}, 
                     {1.0 , 1.0}};
   
    Matrix Y = {{0.0} , {1.0} , {1.0} , {0.0}};

    Dense_layer dense1(2 , 3);
    Tanh tanh1 ;
    Dense_layer dense2 (3, 1);
    Tanh tanh2 ;

    Layer *network[] = {&dense1 , &tanh1 , &dense2 , &tanh2};


    double epochs = 100 ;
    double learning_rate = 0.01 ;
    Vector x ;
    Vector y;

//train
    for (size_t e = 0 ; e < epochs ; ++e){
     double error = 0;

    for (size_t i = 0 ; i < X.size() ; ++i){
         x = X[i]; 
         y = Y[i];  

        //forward
        Vector  output = x ;
        for (auto* layer : network){
             output = layer -> forward(output);
        }

        //error 
        error += mse(y , output); // E .

        //backward
        Vector  grad = mse_prime(y , output);   // dE / dY
        Layer *prev_layer = nullptr ;
        for (auto it = std::rbegin(network) ; it != std::rend(network) ; ++it){
            Layer* layer = *it ;
            grad = layer -> backward( grad , learning_rate);
            prev_layer = layer;
        }
    }

    error /= X.size();  //Average

    std::cout <<  " Epoch : " << e << " Average Error : " << error << std::endl;
    }

    std::cout << "Trainig Done !" << std::endl;
       
       return 0;

}
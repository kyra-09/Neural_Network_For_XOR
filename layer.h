#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <memory>



typedef std::vector<double> Vector;
typedef std::vector<Vector> Matrix;

//abstract class
class Layer 
{
public:
   Layer () = default;
   Layer (const Vector& input) : input(input) {}
   virtual ~Layer () = default;

   //pure virtual functions
   virtual Vector forward (const Vector& input_) = 0;
   virtual Vector backward (  Vector& output_gradient
                            , double learning_rate)  = 0;

protected:
  Vector input ;
  Vector output ;
  double learning_rate = 0.0 ;

};

#endif

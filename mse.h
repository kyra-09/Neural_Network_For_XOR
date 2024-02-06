#include "layer.h"

double mse (const Vector& y_true , const Vector& y_pred)
{
   if (y_true.size() != y_pred.size())
   {
    std::cerr << "Error : size of predicted values and actual values isn't same." << std::endl;
    exit(0);
   }

    double square_diff_sum = 0.00 ;
    for (int i = 0 ; i < y_pred.size() ; ++i){
        double difference = y_pred[i] - y_true[i];
        square_diff_sum = difference * difference;
    }
    return square_diff_sum / y_pred.size();
}

Vector mse_prime (const Vector& y_true , const Vector& y_pred)
{
    if (y_true.size() != y_pred.size())
   {
    std::cerr << "Error : size of predicted values and actual values isn't same." << std::endl;
    exit(0);
   }
    
    Vector output_gradient (y_true.size() , 0.00);
    for (size_t i = 0; i < y_true.size() ; ++i){
        output_gradient[i] = 2 * (y_pred[i] - y_true[i] ) / y_true.size();
    }

    return output_gradient;
}
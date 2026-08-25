#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>


using namespace std;


class Neuron {
    public :
        vector <double> weights;
        double bias;

        Neuron(int input_size){
            weights.resize(input_size, 0.0);
            bias = 0.0;
        }

        // activation function - ReLU
        double relu(double x) {
            return max(x, 0.0);
        }

        // differentiated ReLU { 1 for > 0 & 0 for < 0 }
        double drelu(double x){
            return x>0 ? 1 : 0;
        }

        // this is prediction - also called forward 
        double forward(const vector<double>&inputs){
            double z= bias;
            for(size_t i=0; i<inputs.size(); i++){
                z += inputs[i] * weights[i]; 
            }

             return z;
        }

        // loss calculation

        // squared error -  not mse
        double squared_error(double target, double prediction){
            return (target - prediction) * (target - prediction);
        }

        // gradient for weights
        double gradient(
            double target,
            double prediction,
            double z,
            double input // here the input x(i) for weight w(i)
        ){
            return 2*(prediction - target) * drelu(z) * input;
        }
        // gradient for bias
        double bias_gradient(double target, double prediction, double z) {
           return 2 * (prediction - target) * drelu(z);
        }

        // gradient descent
        double new_weight (double old_weight, double learning_rate, double gradient){
            return old_weight - (learning_rate * gradient);
        }
};

int main(){
    Neuron neuron(3);

    neuron.weights[0] = 0.5;
    neuron.weights[1] = 0.2;
    neuron.weights[2] = 0.8;

    neuron.bias = 1.0;

    vector<double> input = {2, 3, 4};
    size_t input_c = input.size();


    double target = 10.0;

    double learning_rate = 0.01;

    double prediction = 0.0;

    int count = 1;
    while(abs(prediction - target) > 0.001){
        
        cout << "Iteration : "<<count << endl;


        double newWeight1 = neuron.weights[0];
        double newWeight2 = neuron.weights[1];
        double newWeight3 = neuron.weights[2];

        double newBias = neuron.bias;

    double z = neuron.forward(input); // pre-relu (pre-activation)
    prediction = neuron.relu(z);
        
    cout << "Prediction : " << prediction << endl;

    cout << "The Loss : " << neuron.squared_error(target, prediction) << endl;

    double gradient_1 = neuron.gradient(target, prediction, z, input[0]);
    cout << "The Gradient 1 : " << gradient_1 << endl;
    double gradient_2 = neuron.gradient(target, prediction, z, input[1]);
    cout << "The Gradient 2 : " << gradient_2 << endl;
    double gradient_3 = neuron.gradient(target, prediction, z, input[2]);
    cout << "The Gradient 3 : " << gradient_3 << endl;
    double gradient_b = neuron.bias_gradient(target, prediction, z);
    cout << "The Bias Gradient : " << gradient_b << endl;
    
   

    cout << "The new weight from old " << neuron.weights[0] << " is : " ;
    neuron.weights[0] = neuron.new_weight(
        newWeight1,  learning_rate, gradient_1);
        cout << neuron.weights[0] << endl;
    cout << "The new weight from old " << neuron.weights[1] << " is : " ;
    neuron.weights[1] = neuron.new_weight(
        newWeight2,  learning_rate, gradient_2);
        cout << neuron.weights[1] << endl;
    cout << "The new weight from old " << neuron.weights[2] << " is : " ;
    neuron.weights[2] = neuron.new_weight(
        newWeight3,  learning_rate, gradient_3);
        cout << neuron.weights[2] << endl;
    cout << "The new bias from old " << neuron.bias << " is : " ;
    neuron.bias = neuron.new_weight(
    newBias, learning_rate, gradient_b);
        cout << neuron.bias << endl;
    
        count++;
        cout << "------------------------------" << endl;
    }
    
}

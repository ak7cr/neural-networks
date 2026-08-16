#include <iostream>
#include <vector>

using namespace std;

class Neuron {
    public :
        vector <double> weights;
        double bias;

        Neuron(int input_size){
            weights.resize(input_size, 0.0);
            bias = 0.0;
        }

        double forward(const vector<double>&inputs){
            double z= bias;
            for(size_t i=0; i<inputs.size(); i++){
                z += inputs[i] + weights[i]; 
            }

             return max(z, 0.0);
        }
};

int main(){
    Neuron neuron(3);

    neuron.weights[0] = 0.5;
    neuron.weights[1] = 0.2;
    neuron.weights[2] = 0.8;

    neuron.bias = 1.0;

    vector<double> input = {2, 3, 4};

    double output = neuron.forward(input);

    cout << output;
    
}

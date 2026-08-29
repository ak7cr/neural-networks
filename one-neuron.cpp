#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

class Neuron {
public:
    vector<double> weights;
    double bias;

    Neuron(int input_size) {
        weights.resize(input_size, 0.0);
        bias = 0.0;
    }

    // Activation function - ReLU
    double relu(double x) {
        return max(x, 0.0);
    }

    // Derivative of ReLU
    double drelu(double x) {
        return x > 0 ? 1 : 0;
    }

    // Forward pass
    double forward(const vector<double>& inputs) {
        double z = bias;

        for (size_t i = 0; i < inputs.size(); i++) {
            z += inputs[i] * weights[i];
        }

        return z;
    }

    // Squared error
    double squared_error(double target, double prediction) {
        return (target - prediction) * (target - prediction);
    }

    // Gradient for weights
    double gradient(
        double target,
        double prediction,
        double z,
        double input
    ) {
        return 2 * (prediction - target) * drelu(z) * input;
    }

    // Gradient for bias
    double bias_gradient(
        double target,
        double prediction,
        double z
    ) {
        return 2 * (prediction - target) * drelu(z);
    }

    // Gradient descent
    double new_weight(
        double old_weight,
        double learning_rate,
        double gradient
    ) {
        return old_weight - (learning_rate * gradient);
    }
};

int main() {

    Neuron neuron(3);

    neuron.weights[0] = 0.5;
    neuron.weights[1] = 0.2;
    neuron.weights[2] = 0.8;

    neuron.bias = 1.0;

    vector<double> input = {2, 3, 4};

    double target = 10.0;

    double learning_rate = 0.0001;

    double threshold = 0.00001;
    int count = 1;

    while (true) {

        // 1. Forward pass
        double z = neuron.forward(input);
        double prediction = neuron.relu(z);

        // 2. Calculate loss
        double loss = neuron.squared_error(target, prediction);

        // 3. Check stopping condition
        if (loss <= threshold)
            break;

        // 4. Calculate gradients
        double gradient_1 =
            neuron.gradient(target, prediction, z, input[0]);

        double gradient_2 =
            neuron.gradient(target, prediction, z, input[1]);

        double gradient_3 =
            neuron.gradient(target, prediction, z, input[2]);

        double gradient_b =
            neuron.bias_gradient(target, prediction, z);

        // 5. Update weights
        neuron.weights[0] =
            neuron.new_weight(
                neuron.weights[0],
                learning_rate,
                gradient_1
            );

        neuron.weights[1] =
            neuron.new_weight(
                neuron.weights[1],
                learning_rate,
                gradient_2
            );

        neuron.weights[2] =
            neuron.new_weight(
                neuron.weights[2],
                learning_rate,
                gradient_3
            );

        // 6. Update bias
        neuron.bias =
            neuron.new_weight(
                neuron.bias,
                learning_rate,
                gradient_b
            );

        // 7. Print progress
        if (count % 10 == 0) {
            cout << "Iteration: " << count
                 << " | Prediction: " << prediction
                 << " | Loss: " << loss
                 << endl;
        }

        // 8. Next iteration
        count++;
    }

    cout << "\nTraining completed!" << endl;
    cout << "Iterations: " << count << endl;
    cout << "Final prediction: "
         << neuron.relu(neuron.forward(input))
         << endl;

    cout << "Final loss: "
         << neuron.squared_error(
                target,
                neuron.relu(neuron.forward(input))
            )
         << endl;

    return 0;
}

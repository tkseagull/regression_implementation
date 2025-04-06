#include <iostream>
#include <vector>
#include <cmath>
#include "logistic_regression_model.h"  // Include the header for the class definition
using namespace std;

// Constructor definition to initialize member variables
logistic_regression::logistic_regression() {
    learning_rate = 0.0;
    epochs = 0;
    weight = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX)); // Random initialization
    bias = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX)); // Random initialization
    previous_loss = 0.0;
}

// Method to initialize the model with given learning rate and epochs
void logistic_regression::initiateModel(float learning_rate, int epochs) {
    this->learning_rate = learning_rate;
    this->epochs = epochs;
}

// Method to display the current model parameters
void logistic_regression::displayModel() {
    cout << "Learning Rate: " << learning_rate << endl;
    cout << "Epochs: " << epochs << endl;
}

// Method for calculating the y predicted value by first using a weight and bias to calculate log odds (z) and then applying the sigmoid function
float logistic_regression::calc_y(float x) {
    float z = weight * x + bias;
    // Clip z to prevent overflow
	z = std::max(std::min(z, 500.0f), -500.0f); // Clipping since its a sigmoid function that can go to infinity appraching 1 and -1. Same result but a bit faster and less chance of error with large numbers.
    return 1 / (1 + exp(-z));
}


float logistic_regression::log_loss(const std::vector<float>& y_pred, const std::vector<float>& y_actual) {
    float log_loss = 0.0;
	
    for (int i = 0; i < y_pred.size(); ++i) {
        
        // Clip predictions to avoid log(0)
		float p = std::max(std::min(y_pred[i], 1.0f - 1e-15f), 1e-15f); // Clipping to avoid log(0) which is undefined
        log_loss += y_actual[i] * std::log(p) + (1 - y_actual[i]) * std::log(1 - p);
    }

    return -log_loss / y_pred.size();
}

// Takes in sample data, X values and corresponding Y values, and calculates the gradients for weight and bias
void logistic_regression::gradientDescent(const std::vector<float>& X, const std::vector<float>& Y) {
    int N = X.size();
    float dW = 0.0; // Initialize the gradient of the weight to zero. Another common method is to set these to random values
    float db = 0.0;

    // Calculate gradients
    for (int i = 0; i < N; ++i) {
        float prediction = calc_y(X[i]); // Plug in curent weight in bias into formula to get prediction
        float error = prediction - Y[i]; // Calculate loss of prediction of individual data point

        // First we sum each predicted value minus the actual value and then multiply it by two. 
        // Then we divide the sum by the number of examples.The result is the slope of the line tangent to the value of the bias.
        dW += error * X[i];
        db += error;
    }

    // Update weights and bias
    weight -= (learning_rate * dW) / N;
    bias -= (learning_rate * db) / N;
}

bool logistic_regression::isConverged(const std::vector<float>& y_pred, const std::vector<float>& Y) {
    double threshold = 1e-7; // Since the number is big a double makes more sense when it comes to accuracy
    return abs(previous_loss - log_loss(y_pred, Y)) < threshold;
}


// Method to train the model with provided data
void logistic_regression::fit(const std::vector<float>& X, const std::vector<float>& Y) {
    std::vector<float> y_pred;

    // Initialize previous_error to a large value or the first computed error
    previous_loss = std::numeric_limits<float>::max(); // or set to 0 if appropriate

    int epoch = 0; // Initialize epoch counter

    // Train the model until convergence or maximum epochs
    while (epoch < epochs) {
        // Perform Gradient Descent
        gradientDescent(X, Y);

        // Update predictions
        y_pred.clear();
        for (auto& x : X) {
            y_pred.push_back(calc_y(x));
        }

        // Calculate the current error
        float loss = log_loss(y_pred, Y);

        // Optionally track progress (e.g., print MSE every 5 epochs)
        if (epoch % 5 == 0) {
            cout << "Epoch " << epoch << " - Log Loss: " << loss << endl;
        }

        // Check for convergence
        if (isConverged(y_pred, Y)) {
            cout << "Converged after " << epoch << " epochs." << endl;
            break; // Exit the loop if converged
        }

        // Update previous_error for the next iteration
        previous_loss = loss;

        // Increment the epoch counter
        epoch++;
    }
}


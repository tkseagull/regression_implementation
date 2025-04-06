#include <iostream>
#include <vector>
#include <cmath>
#include "linear_regression_model.h"  // Include the header for the class definition
using namespace std;

// Constructor definition to initialize member variables
linear_regression::linear_regression() {
    learning_rate = 0.0;
    epochs = 0;
    weight = 0.0;
    bias = 0.0;
    previous_error = 0.0;
}

// Method to initialize the model with given learning rate and epochs
void linear_regression::initiateModel(float learning_rate, int epochs) {
    this->learning_rate = learning_rate;
    this->epochs = epochs;
}

// Method to display the current model parameters
void linear_regression::displayModel() {
    cout << "Learning Rate: " << learning_rate << endl;
    cout << "Epochs: " << epochs << endl;
}

// Method for calculating the y predicted value using a weight and bias
float linear_regression::calc_y(float x) {
	return weight * x + bias; // y = wx + b
}

// Method to calculate the MSE error (avg value of squared differences between predicted and actual values)
// Takes in vectors of predicted and actual values and returns subtracts the y_pred from y_actual and squares the result
float linear_regression::mse_error(const std::vector<float>& y_pred, const std::vector<float>& y_actual) {
    float mse_error = 0.0;
    for (int i = 0; i < y_pred.size(); i++) {
		mse_error += (y_pred[i] - y_actual[i]) * (y_pred[i] - y_actual[i]);  // Squaring the error by multiplying it by itself
    }
	return mse_error / y_actual.size();  // Return the average of the squared errors
}

// Takes in sample data, X values and corresponding Y values, and calculates the gradients for weight and bias
void linear_regression::gradientDescent(const std::vector<float>& X, const std::vector<float>& Y) {
    int N = X.size();
	float dW = 0.0; // Initialize the gradient of the weight to zero. Another common method is to set these to random values
    float db = 0.0;

    // Calculate gradients
    for (int i = 0; i < N; ++i) {
		float prediction = calc_y(X[i]); // Plug in curent weight in bias into formula to get prediction
		float error = prediction - Y[i]; // Calculate loss of prediction OF individual data point
        
        // First we sum each predicted value minus the actual value and then multiply it by two. 
        // Then we divide the sum by the number of examples.The result is the slope of the line tangent to the value of the bias.
        dW += error * X[i]; 
        db += error;
    }

    // Update weights and bias
    weight -= (learning_rate * 2 * dW) / N;
    bias -= (learning_rate * 2 * db) / N;
}

bool linear_regression::isConverged(const std::vector<float>& y_pred, const std::vector<float>& Y) {
    double threshold = 1e-7;
    return abs(previous_error - mse_error(y_pred, Y)) < threshold;
}

void linear_regression::fit(const std::vector<float>& X, const std::vector<float>& Y) {

    std::vector<float> y_pred;

    // Initialize previous_error to a large value or the first computed error
    previous_error = std::numeric_limits<float>::max(); // or set to 0 if appropriate

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
        float error = mse_error(y_pred, Y);

        // Optionally track progress (e.g., print MSE every 5 epochs)
        if (epoch % 5 == 0) {
            cout << "Epoch " << epoch << " - MSE: " << error << endl;
        }

        // Check for convergence
        if (isConverged(y_pred, Y)) {
            cout << "Converged after " << epoch << " epochs." << endl;
            break; // Exit the loop if converged
        }

        // Update previous_error for the next iteration
        previous_error = error;

        // Increment the epoch counter
        epoch++;
    }
}



#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include <vector> // Include the vector header

class logistic_regression {
private:
    float learning_rate;
    int epochs;
    float weight;
    float bias;
    float previous_loss;
    std::vector<float> y_pred; // To store predictions

public:
    // Constructor to initialize member variables
    logistic_regression();

    // Method to initialize the model with given learning rate and epochs
    void initiateModel(float learning_rate, int epochs);

    // Method to display the current model parameters
    void displayModel();

    // Gradient Descent Method to update the weights and bias
    void gradientDescent(const std::vector<float>& X, const std::vector<float>& Y);

    // Method to calculate the y predicted value using a weight and bias
    float calc_y(float x);

    // Method to calculate the MSE error (avg value of squared differences between predicted and actual values)
    float log_loss(const std::vector<float>& y_pred, const std::vector<float>& y_actual);

    // Method to train the model with provided data
    void fit(const std::vector<float>& X, const std::vector<float>& Y);

    bool isConverged(const std::vector<float>& X, const std::vector<float>& Y);

    float getWeight() { return weight; }
    float getBias() { return bias; }
};

#endif // LINEAR_REGRESSION_H
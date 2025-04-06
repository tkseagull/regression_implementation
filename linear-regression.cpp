// linear-regression.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "linear_regression_model.h"
#include "logistic_regression_model.h"
#include <vector>

int main() {

    float model;
    std::cout << "Enter 0 to test the linear regression model or 1 for logistic regression model: ";
    std::cin >> model;

	if (model == 0) {
		linear_regression linear;
		linear.initiateModel(0.01, 10000);  // Initialize with learning rate 0.01 and 1000 epochs
		// Create sample data
		std::vector<float> X = { 1, 2, 3, 4, 5 };
		std::vector<float> Y = { 2, 3, 5, 7, 11 };
		linear.fit(X, Y);  // Train the model
		// Make predictions on new data
		float newX;
		std::cout << "Enter a new x value to predict y: ";
		std::cin >> newX;
		float predictedY = linear.calc_y(newX);
		std::cout << "Line of best fit: y = " << linear.getWeight() << "x + " << linear.getBias() << std::endl;
		std::cout << "Predicted y-value for x = " << newX << ": " << predictedY << std::endl;
	}
	else if (model == 1) {
		logistic_regression logistic;
		// Initialize the model with a learning rate and number of epochs
		logistic.initiateModel(0.001, 10000);  // Example: learning rate = 0.01, epochs = 10000
		// Create sample data
		std::vector<float> X = { 0.0, 1.0, 2.0, 3.0, 4.0 }; // Example feature values
		std::vector<float> Y = { 0.0, 0.0, 0.0, 1.0, 1.0 }; // Corresponding labels (0 or 1)
		// Train the model
		logistic.fit(X, Y);
		
		// Make predictions on new data
		float newX;
		std::cout << "Enter a new x value to predict y (probability): ";
		std::cin >> newX;
		float predictedY = logistic.calc_y(newX); // Get the predicted probability
		std::cout << "Logistic regression equation: P(y=1 | x) = 1 / (1 + exp(-(" << logistic.getWeight() << " * " << newX << " + " << logistic.getBias() << ")))" << std::endl;
		std::cout << "Predicted probability for x = " << newX << ": " << predictedY << std::endl;
	}
    
	return 0;

}

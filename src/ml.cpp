#include "../include/ml_cpp/ml.h"
#include "../include/ml_cpp/math_utilities.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

namespace ml {

    /**
     * @brief Fits the regression model to the given training data.
     *
     * This function uses gradient descent to optimize the model weights and bias
     * based on the provided training data (X and y), learning rate, and number of iterations.
     *
     * @param X A vector of vectors containing the feature data for training.
     * @param y A vector containing the target values for training.
     * @param learning_rate The learning rate for the gradient descent optimization.
     * @param iterations The number of iterations to run the gradient descent algorithm (default is 100).
     */
    void Regression::fit(
        const vector<vector<float>>& X,
        const vector<float>& y, 
        float learning_rate,
        int iterations = 100
    ) {
        size_t num_features = X.at(0).size();
        
        model_weights = vector<float>(num_features, 0.0f);

        GradientDescentResult result = gradient_descent(
            X, 
            y, 
            model_weights, 
            model_bias, 
            learning_rate, 
            iterations
        );

        model_weights = result.optimized_weights;
        model_bias = result.optimized_bias;
    }

    /**
     * @brief Computes the gradient of the loss function with respect to the weights and bias for a regression model.
     *
     * This function calculates the gradients of the loss function with respect to the model parameters (weights and bias)
     * using the provided input features and target values. The gradients are used in optimization algorithms to update
     * the model parameters and minimize the loss function.
     *
     * @param X A 2D vector of input features, where each inner vector represents a single example.
     * @param y A vector of target values corresponding to each example in the input features.
     * @param weights A vector of current weights of the regression model.
     * @param bias The current bias of the regression model.
     * @return A ComputeGradientResult struct containing the gradients of the weights and the bias.
     */
    ComputeGradientResult Regression::compute_gradient(
        const vector<vector<float>>& X, 
        const vector<float>& y, 
        const vector<float>& weights, 
        float bias   
    ) const {
        size_t num_examples = X.size();
        size_t num_features = X.at(0).size();

        vector<float> grad_w(num_features, 0.0f);
        float grad_b = 0.0f;

        for (size_t i = 0; i < num_examples; ++i) {
            float prediction = predict(X[i], weights, bias);

            float error = prediction - y[i];

            for (size_t j = 0; j < X[i].size(); ++j) {
                grad_w[j] += error * X[i][j];
            }

            grad_b += error;
        }

        for (size_t j = 0; j < grad_w.size(); ++j) {
            grad_w[j] /= num_examples;
        }
        grad_b /= num_examples;

        return { grad_w, grad_b };           
    };

    /**
     * @brief Performs gradient descent optimization for linear regression.
     *
     * This function iteratively updates the weights and bias to minimize the cost function
     * using the gradient descent algorithm.
     *
     * @param X A 2D vector of floats representing the feature matrix.
     * @param y A vector of floats representing the target values.
     * @param initial_weights A vector of floats representing the initial weights.
     * @param initial_bias A float representing the initial bias.
     * @param learning_rate A float representing the learning rate for gradient descent.
     * @param iterations An integer representing the number of iterations to perform.
     * @return GradientDescentResult A struct containing the optimized weights, bias, cost history, and parameters history.
     *
     * The function prints the cost, weights, and bias at regular intervals during the iterations.
     */
    GradientDescentResult Regression::gradient_descent(
        const vector<vector<float>>& X, 
        const vector<float>& y, 
        vector<float> initial_weights, 
        float initial_bias, 
        float learning_rate, 
        int iterations
    ) const {
        size_t num_features = X.at(0).size();

        vector<float> cost_history(iterations, 0.0f);
        vector<vector<float>> parameters_history(iterations, vector<float>(num_features + 1, 0.0f)); // Store all weights and bias

        vector<float> weights = initial_weights;
        float bias = initial_bias;

        for (int i = 0; i < iterations; ++i) {
            ComputeGradientResult gradient = compute_gradient(X, y, weights, bias);

            for (size_t j = 0; j < num_features; ++j) {
                weights[j] -= learning_rate * gradient.weights[j];
            }
            bias -= learning_rate * gradient.bias;

            cost_history[i] = compute_cost(X, y, weights, bias);

            parameters_history[i] = weights;
            parameters_history[i].push_back(bias);

            if (i % ((iterations + 9) / 10) == 0) {
                cout << "Iteration: " << setw(4) << i 
                    << " | Cost: " << fixed << setprecision(6) << cost_history[i]
                    << " | Weights: ";
                for (float w : weights) {
                    cout << fixed << setprecision(4) << w << " ";
                }
                cout << " | Bias: " << fixed << setprecision(4) << bias 
                    << endl;
            }
        }

        return {weights, bias, cost_history, parameters_history};
    }

    /**
     * @brief Predicts the output for a given input vector using the linear regression model.
     *
     * This function calculates the predicted value by computing the dot product of the input vector
     * and the weights vector, and then adding the bias term.
     *
     * @param x The input vector for which the prediction is to be made.
     * @param weights The vector of weights corresponding to the input features.
     * @param bias The bias term to be added to the dot product of weights and input vector.
     * @return The predicted value as a float.
     */
    float LinearRegression::predict(
        const vector<float>& x,
        const vector<float>& weights,
        float bias
    ) const {
        float prediction = bias;

        prediction += dot_product(weights, x);

        return prediction;   
    }

    /**
     * @brief Predicts the output for a given input vector using the linear regression model.
     * 
     * This function takes a vector of input features and returns the predicted output
     * based on the model's weights and bias.
     * 
     * @param x A vector of floats representing the input features.
     * @return A float representing the predicted output.
     */
    float LinearRegression::predict(
        const vector<float>& x
    ) const {
        return predict(x, model_weights, model_bias);
    }

    /**
     * @brief Computes the cost for linear regression.
     *
     * This function calculates the mean squared error cost for a given set of 
     * features, target values, weights, and bias. It is used to evaluate the 
     * performance of the linear regression model.
     *
     * @param X A 2D vector of floats representing the feature matrix, where each 
     *          inner vector corresponds to a single example.
     * @param y A vector of floats representing the target values.
     * @param weights A vector of floats representing the weights of the linear 
     *                regression model.
     * @param bias A float representing the bias term of the linear regression model.
     * @return The computed cost as a float.
     */
    float LinearRegression::compute_cost(
        const vector<vector<float>>& X, 
        const vector<float>& y, 
        const vector<float>& weights, 
        float bias
    ) const {
        size_t num_examples = X.size();
        size_t num_features = X.at(0).size();

        float cost = 0.0f;

        for (size_t i = 0; i < num_examples; ++i) {
            float prediction = bias;
            
            for (size_t j = 0; j < num_features; ++j) {
                prediction += weights[j] * X[i][j];
            }

            cost += (prediction - y[i]) * (prediction - y[i]);;
        }

        cost /= (2 * num_examples);
        return cost;
    }

    /**
     * @brief Computes the cost (loss) for the linear regression model.
     *
     * This function calculates the cost using the provided feature matrix X and 
     * target vector y. It uses the model's weights and bias to compute the cost.
     *
     * @param X A 2D vector of floats representing the feature matrix.
     * @param y A vector of floats representing the target values.
     * @return The computed cost as a float.
     */
    float LinearRegression::compute_cost(
        const vector<vector<float>>& X, 
        const vector<float>& y
    ) const { 
        return LinearRegression::compute_cost(X, y, model_weights, model_bias);
    }

    /**
     * @brief Predicts the probability of a binary outcome using logistic regression.
     *
     * This function computes the logit (linear combination of input features and weights plus bias),
     * applies the sigmoid function to obtain the probability of the binary outcome.
     *
     * @param x A vector of input features.
     * @param weights A vector of weights corresponding to the input features.
     * @param bias The bias term.
     * @return The predicted probability of the binary outcome.
     */
    float LogisticRegression::predict(
        const vector<float>& x,
        const vector<float>& weights,
        float bias
    ) const {
        size_t num_features = x.size(); 
        
        float logit = bias;

        logit += dot_product(weights, x);

        float prediction = sigmoid(logit);

        return prediction;
    }

    /**
     * @brief Predicts the probability of the positive class for a given input vector.
     *
     * This method takes an input vector `x` and uses the model's weights and bias to
     * compute the probability of the positive class using the logistic regression model.
     *
     * @param x A vector of floats representing the input features.
     * @return A float representing the predicted probability of the positive class.
     */
    float LogisticRegression::predict(
        const vector<float>& x
    ) const {
        return predict(x, model_weights, model_bias);
    }

    /**
     * @brief Computes the cost for logistic regression.
     * 
     * This function calculates the cost (or loss) for logistic regression using 
     * the given feature matrix, labels, weights, and bias. The cost is computed 
     * using the logistic loss function.
     * 
     * @param X A 2D vector of floats representing the feature matrix, where each 
     *          inner vector corresponds to a single example.
     * @param y A vector of floats representing the true labels for each example.
     * @param weights A vector of floats representing the weights for each feature.
     * @param bias A float representing the bias term.
     * @return A float representing the computed cost.
     */
    float LogisticRegression::compute_cost(
        const vector<vector<float>>& X, 
        const vector<float>& y, 
        const vector<float>& weights, 
        float bias
    ) const {
        size_t num_examples = X.size();
        size_t num_features = X.at(0).size();

        float cost = 0.0;

        for (size_t i = 0; i < num_examples; i++) {
            float logit = bias;

            for (size_t j = 0; j < num_features; j++) {
                logit += weights[j] * X[i][j];
            }

            float prediction = sigmoid(logit);

            cost += (-y[i] * log(prediction)) - ((1 - y[i]) * log(1 - prediction));
        }

        cost /= (2 * num_examples);
        return cost;
    }

    /**
     * @brief Computes the cost for the logistic regression model.
     * 
     * This function calculates the cost (or loss) of the logistic regression model
     * using the provided feature matrix X and the target vector y. It utilizes the
     * model's weights and bias to compute the cost.
     * 
     * @param X A 2D vector of floats representing the feature matrix.
     * @param y A vector of floats representing the target values.
     * @return The computed cost as a float.
     */
    float LogisticRegression::compute_cost(
        const vector<vector<float>>& X, 
        const vector<float>& y
    ) const { 
        return LogisticRegression::compute_cost(X, y, model_weights, model_bias);
    }
}

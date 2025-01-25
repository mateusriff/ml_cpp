#include "../include/ml_cpp/ml.h"
#include "../include/ml_cpp/math_utilities.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

namespace ml {

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

    float LinearRegression::predict(
        const vector<float>& x,
        const vector<float>& weights,
        float bias
    ) const {
        float prediction = bias;

        prediction += dot_product(weights, x);

        return prediction;   
    }

    float LinearRegression::predict(
        const vector<float>& x
    ) const {
        return predict(x, model_weights, model_bias);
    }

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

    float LinearRegression::compute_cost(
        const vector<vector<float>>& X, 
        const vector<float>& y
    ) const { 
        return LinearRegression::compute_cost(X, y, model_weights, model_bias);
    }

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

    float LogisticRegression::predict(
        const vector<float>& x
    ) const {
        return predict(x, model_weights, model_bias);
    }

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

    float LogisticRegression::compute_cost(
        const vector<vector<float>>& X, 
        const vector<float>& y
    ) const { 
        return LogisticRegression::compute_cost(X, y, model_weights, model_bias);
    }
}

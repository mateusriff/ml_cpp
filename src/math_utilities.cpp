#include <cmath>
#include <vector>
#include <stdexcept>

using namespace std;

/**
 * @brief Computes the sigmoid function for a given input.
 *
 * The sigmoid function is defined as 1 / (1 + exp(-x)), where exp is the exponential function.
 * It is commonly used in machine learning and neural networks as an activation function.
 *
 * @param x The input value for which the sigmoid function is to be computed.
 * @return The computed sigmoid value.
 */
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

/**
 * @brief Computes the dot product of two vectors.
 *
 * This function calculates the dot product of two vectors of floats.
 * The vectors must have the same size, otherwise an invalid_argument
 * exception is thrown.
 *
 * @param a The first vector.
 * @param b The second vector.
 * @return The dot product of the two vectors.
 * @throws std::invalid_argument if the vectors do not have the same size.
 */
float dot_product(const vector<float>& a, const vector<float>& b) {
    if (a.size() != b.size()) {
        throw invalid_argument("Vectors must have the same size for dot product.");
    }

    float result = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

/**
 * @brief Expands the features of the input matrix X to a specified polynomial degree.
 *
 * This function takes a matrix of input features and expands each feature to the specified polynomial degree.
 * The expanded features include all powers of the original features from 0 up to the specified degree.
 *
 * @param X A 2D vector of floats representing the input features, where each inner vector is a sample.
 * @param degree An integer representing the maximum degree of the polynomial expansion.
 * @return A 2D vector of floats representing the polynomially expanded features.
 *
 * @note The resulting vector will have a size of (num_examples x (num_features * (degree + 1))).
 */
vector<vector<float>> polynomial_expansion(const vector<vector<float>>& X, int degree) {
    vector<vector<float>> X_poly;

    size_t num_examples = X.size();
    size_t num_features = X.at(0).size();

    for (size_t i = 0; i < num_examples; ++i) {
        vector<float> transformedSample;

        for (int d = 0; d <= degree; ++d) {
            for (size_t j = 0; j < num_features; ++j) {
                transformedSample.push_back(pow(X.at(i).at(j), d));
            }
        }
        X_poly.push_back(transformedSample);
    }
    
    return X_poly;
}
#include <cmath>
#include <vector>
#include <stdexcept>

using namespace std;

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

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
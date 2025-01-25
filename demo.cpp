#include <iostream>
#include <vector>
#include "include/ml_cpp/ml.h"
#include "include/ml_cpp/math_utilities.h"

using namespace std;

void train_and_evaluate_linear_regression() {
    // Training set
    vector<vector<float>> X_train = {
        {1.03f, 2.05f}, {2.01f, 3.02f}, {2.97f, 3.95f}, {4.02f, 5.01f},
        {5.05f, 6.04f}, {6.00f, 7.01f}, {6.98f, 8.03f}, {8.02f, 9.00f},
        {8.95f, 10.05f}, {0.98f, 1.99f}
    };
    vector<float> y_train = {6.1f, 9.9f, 13.7f, 18.3f, 22.1f, 26.2f, 30.1f, 34.2f, 38.0f, 5.7f};

    // Testing set
    vector<vector<float>> X_test = {
        {1.10f, 2.08f}, {3.00f, 3.98f}, {4.10f, 5.05f}, {5.08f, 6.10f}, {7.05f, 8.02f}
    };
    vector<float> y_test = {6.3f, 14.0f, 18.5f, 22.4f, 30.5f};

    // Training 
    ml::LinearRegression model;
    model.fit(X_train, y_train, 0.01f, 500);

    // Evaluating
    float mse = model.compute_cost(X_test, y_test);
    cout << "Linear Regression Mean Squared Error on test set: " << mse << endl;
}

void train_and_evaluate_logistic_regression() {
    // Training set
    vector<vector<float>> X_train = {
        {1.03f, 2.05f}, {2.01f, 3.02f}, {2.97f, 3.95f}, {4.02f, 5.01f},
        {5.05f, 6.04f}, {6.00f, 7.01f}, {6.98f, 8.03f}, {8.02f, 9.00f},
        {8.95f, 10.05f}, {0.98f, 1.99f}
    };
    vector<float> y_train = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f};

    // Testing set
    vector<vector<float>> X_test = {
        {1.10f, 2.08f}, {3.00f, 3.98f}, {4.10f, 5.05f}, {5.08f, 6.10f}, {7.05f, 8.02f}
    };
    vector<float> y_test = {0.0f, 0.0f, 1.0f, 1.0f, 1.0f};

    // Training 
    ml::LogisticRegression model;
    model.fit(X_train, y_train, 0.01f, 500);

    // Evaluating
    float mse = model.compute_cost(X_test, y_test);
    cout << "Logistic Regression Mean Squared Error on test set: " << mse << endl;
}

int main() {
    cout << "Training and evaluating Linear Regression model..." << endl;
    train_and_evaluate_linear_regression();

    cout << "Training and evaluating Logistic Regression model..." << endl;
    train_and_evaluate_logistic_regression();

    return 0;
}
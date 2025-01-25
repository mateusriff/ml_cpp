#ifndef MATH_UTILITIES_H
#define MATH_UTILITIES_H

#include <vector>
#include <cmath>

using namespace std;

float sigmoid(float x);

float dot_product(const vector<float>& a, const vector<float>& b);

vector<vector<float>> polynomial_expansion(const vector<vector<float>>& X, int degree);


#endif
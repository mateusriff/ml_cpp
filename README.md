# ml_cpp

A basic Machine Learning library implemented from scratch in C++ for my Object Oriented Data Structures course assignment. Features include Linear Regression and Logistic Regression.

## Structure and UML Diagram

The library is composed of three classes, one being an abstract base class `Regression` from which the other two, `LinearRegression` and `LogisticRegression`, are derived. Two structs define the custom types of values returned by some function members.

![uml-diagram](https://github.com/user-attachments/assets/cf971a38-01d8-496c-8c12-4bd98fc876ba)

Object Oriented Programming principles were deliberately not applied to `demo.cpp` for clarity's sake.

## Usage

To use this library, first clone this repository:

```bash
git clone https://github.com/mateusriff/ml_cpp.git
cd ml_cpp
```

If you want to use this library in another C++ project of yours, build the library with CMake (make sure you have it installed) by running:

```bash
mkdir build
cd build
cmake ..
make
```

Then, to link the library to your project, add this to `CMakeLists.txt`:

```txt
# Add the path to the ml_cpp library
add_subdirectory(/path/to/ml_cpp/build)

# ...

# Link the ml_cpp library
target_link_libraries(your_project PRIVATE ml_lib)
```

### Demo

Let's fit a Linear Regression model to some data:

```cpp
#include <vector>
#include "ml_lib/ml.h"

int main() {
    // Set the training data
    vector<vector<float>> X = {
        {1.0f, 2.0f},
        {2.0f, 3.0f},
        {3.0f, 4.0f},
        {4.0f, 5.0f} 
    };
    vector<float> y = {6.0f, 10.0f, 14.0f, 18.0f};

    // Instantiate a Linear Regression model
    ml::LinearRegression model;

    // Train the model
    float learning_rate = 0.01f;
    int iterations = 1000;
    model.fit(X, y, 0.01f, 1000);

    // Test the model
    vector<float> testPoint = {5.0f, 6.0f};
    float prediction = model.predict(testPoint);

    cout << "Prediction for {5.0f, 6.0f}: " << prediction << endl; // predicted target value should be around 22

    return 0;
}
```

While training, a log is printed to our console:
```bash
Iteration:    0 | Cost: 50.125053 | Weights: 0.3500 0.4700  | Bias: 0.1200
Iteration:  100 | Cost: 0.029779 | Weights: 1.6640 2.1298  | Bias: 0.4658
Iteration:  200 | Cost: 0.021114 | Weights: 1.7171 2.1093  | Bias: 0.3922
.
.
.
Iteration:  900 | Cost: 0.001902 | Weights: 1.9151 2.0328  | Bias: 0.1177
```

For Logistic Regression, we do pretty much the same thing:
```cpp
int main() {
    // Set the training data
    vector<vector<float>> X = {
        {1.0f, 2.0f},
        {2.0f, 1.0f},
        {3.0f, 4.0f},
        {5.0f, 7.0f}
    };
    vector<float> y = {0.0f, 0.0f, 1.0f, 1.0f};

    // Instantiate a Logistic Regression model
    ml::LogisticRegression model;

    // Train the model
    float learning_rate = 0.01f;
    int iterations = 1000;
    model.fit(X, y, learning_rate, iterations);
    
    // Test the model
    vector<float> testPoint = {1.0f, 1.0f};
    float prediction = model.predict(testPoint);

    cout << "Prediction for {1.0f, 1.0f}: " << prediction << endl; // should be closer to 0 than 1

    return 0;
}
```

## Evaluation

The performance of the models I implemented was compared to that of `scikit-learn`, a widely used Python library for Machine Learning. Specifically, the implementations of Linear Regression and Logistic Regression from both libraries were trained on the same training sets for `500` iterations, with a learning rate of `0.01`, and evaluated on the same test sets. The training and test sets are provided in the appendix.

The performance in terms of cost for `scikit-learn`'s implementations was measured in a Jupyter Notebook, which can be accessed [here](https://colab.research.google.com/drive/1nPkp6V4EU-jsifDBsv5RXEyJSbc1wO7o?usp=sharing).


### Results

- The Mean Squared Error (MSE) for the Linear Regression model I implemented was `0.0124`, while the `scikit-learn` implementation had an MSE of approximately `0.05`.

- For Logistic Regression models, the MSE for my implementation was `0.2168`, whereas `scikit-learn`'s implementation was around `0.025`.

The Linear Regression implementation in `ml_cpp` achieved one-fifth of the cost of `scikit-learn`'s implementation, both using the same number of Gradient Descent iterations and the same learning rate. On the other hand, the `scikit-learn` Logistic Regression implementation outperformed mine by an order of magnitude.


## Appendix
Below are the training and testing sets used for comparing the regression and classification implementations.

### Regression Sets

#### Training Set

| X1   | X2   | y    |
|------|------|------|
| 1.03 | 2.05 | 6.1  |
| 2.01 | 3.02 | 9.9  |
| 2.97 | 3.95 | 13.7 |
| 4.02 | 5.01 | 18.3 |
| 5.05 | 6.04 | 22.1 |
| 6.00 | 7.01 | 26.2 |
| 6.98 | 8.03 | 30.1 |
| 8.02 | 9.00 | 34.2 |
| 8.95 | 10.05| 38.0 |
| 0.98 | 1.99 | 5.7  |

### Testing Set

| X1   | X2   | y    |
|------|------|------|
| 1.10 | 2.08 | 6.3  |
| 3.00 | 3.98 | 14.0 |
| 4.10 | 5.05 | 18.5 |
| 5.08 | 6.10 | 22.4 |
| 7.05 | 8.02 | 30.5 |

### Training Sets

#### Training Set

| X1   | X2   | y    |
|------|------|------|
| 1.03 | 2.05 | 0    |
| 2.01 | 3.02 | 0    |
| 2.97 | 3.95 | 0    |
| 4.02 | 5.01 | 1    |
| 5.05 | 6.04 | 1    |
| 6.00 | 7.01 | 1    |
| 6.98 | 8.03 | 1    |
| 8.02 | 9.00 | 1    |
| 8.95 | 10.05| 1    |
| 0.98 | 1.99 | 0    |

#### Testing Set

| X1   | X2   | y    |
|------|------|------|
| 1.10 | 2.08 | 0    |
| 3.00 | 3.98 | 0    |
| 4.10 | 5.05 | 1    |
| 5.08 | 6.10 | 1    |
| 7.05 | 8.02 | 1    |

## Addendum

Building all of this from scratch would not have been possible without the excellent resources from the course [Supervised Machine Learning: Regression and Classification](https://www.coursera.org/learn/machine-learning/), taught by Andrew Ng.
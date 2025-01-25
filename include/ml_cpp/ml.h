#ifndef ML_H
#define ML_H

#include <vector>

using namespace std;

namespace ml {

    struct GradientDescentResult {
        vector<float> optimized_weights;
        float optimized_bias;
        vector<float> cost_history;
        vector<vector<float>> parameters_history;
    };

    struct ComputeGradientResult {
        vector<float> weights;
        float bias;
    };

    class Regression {
    public: 
        Regression(): model_weights(), model_bias(0.0f) {};

        virtual void fit(
            const vector<vector<float>>& X, 
            const vector<float>& y, 
            float learning_rate, 
            int iterations
        );
        virtual float predict(
            const vector<float>& x,
            const vector<float>& weights,
            float bias
        ) const = 0;

    protected:
        vector<float> model_weights;
        float model_bias;
        
        virtual float compute_cost(
            const vector<vector<float>>& X, 
            const vector<float>& y, 
            const vector<float>& weights, 
            float bias      
        ) const = 0;

        virtual ComputeGradientResult compute_gradient(
            const vector<vector<float>>& X, 
            const vector<float>& y, 
            const vector<float>& weights, 
            float bias            
        ) const;

        GradientDescentResult gradient_descent(
            const vector<vector<float>>& X, 
            const vector<float>& y, 
            vector<float> initial_weights, 
            float initial_bias, 
            float learning_rate, 
            int iterations     
        ) const;
    };

    class LinearRegression : public Regression {
    public:
        LinearRegression() {};

        float predict(
            const vector<float>& x,
            const vector<float>& weights,
            float bias
        ) const override;

        float predict(const vector<float>& x) const;

        float compute_cost(
            const vector<vector<float>>& X, 
            const vector<float>& y, 
            const vector<float>& weights, 
            float bias
        ) const override;

        float compute_cost(
            const vector<vector<float>>& X, 
            const vector<float>& y
        ) const;
    };

    class LogisticRegression : public Regression {
    public:
        LogisticRegression() {};

        float predict(
            const vector<float>& x,
            const vector<float>& weights,
            float bias
        ) const override;

        float predict(const vector<float>& x) const;

        float compute_cost(
            const vector<vector<float>>& X, 
            const vector<float>& y, 
            const vector<float>& weights, 
            float bias
        ) const override;

        float compute_cost(
            const vector<vector<float>>& X, 
            const vector<float>& y
        ) const;
    };
}

#endif
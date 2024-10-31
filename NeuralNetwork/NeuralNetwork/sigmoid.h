#pragma once
#include <Eigen/Core>

class Sigmoid {
private:
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;

public:

	static inline void sigmoid(const Matrix& Z, Matrix& activation) {
		activation = Z.unaryExpr([](double x) {
			return (1.0 / (1.0 + exp(-x)));
		});
	}

	static inline void sigmoid_derivative(const Matrix& activation, Matrix& G) {
		G = activation.unaryExpr([](double x) {
			return x * (1.0 - x);
		});
	}
};
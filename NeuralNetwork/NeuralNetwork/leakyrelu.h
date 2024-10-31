#pragma once
#include <Eigen/Core>

class LeakyReLu {
private:
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;

public:
	static inline void leakyrelu(const Matrix& Z, Matrix& activation) {
		activation = Z.unaryExpr([](double x) {
			return std::max(0.01 * x, x);
		});
	}

	static inline void leakyrelu_derivative(const Matrix& activation, Matrix& G) {
		G = activation.unaryExpr([](double x) {
			if (x > 0) return 1.0;

			return 0.01;
		});
	}
};
#pragma once
#include <Eigen/Core>

class ReLu {
private:
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;

public:
	static inline void relu(const Matrix& Z, Matrix& activation) {
		activation = Z.unaryExpr([](double x) {
			return std::max(0.0, x);
		});
	}

	static inline void relu_derivative(const Matrix& Z, Matrix& G) {
		G = Z.unaryExpr([](double x) {
			if (x > 0.0) return 1.0;
			return 0.0;
		});
	}
};
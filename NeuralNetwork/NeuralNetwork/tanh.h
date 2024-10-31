#pragma once 
#include <Eigen/Core>
#include <math.h>
class Tanh {
private:
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;

public:
	static inline void tan_h(const Matrix& Z, Matrix& activation) {
		activation = Z.unaryExpr([](double x) {
			return 2.0 / (1.0 + exp(-2 * x)) - 1;
		});
	}

	static inline void tan_h_derivative(const Matrix& activation, Matrix& G) {
		G = activation.unaryExpr([](double x) {
			return 1 - (x * x);
		});
	}
};
#pragma once
#include <Eigen/Core>

class Linear {
private:
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;

public:
	static inline void linear(Matrix& activation, const Matrix& Z) {
		activation = Z;
	}

	static inline void linear_derivative(const Matrix& activation, Matrix& G) {
		G = Matrix::Ones(activation.rows(), activation.cols());
	}
};


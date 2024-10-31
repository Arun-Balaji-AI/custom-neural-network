#pragma once
#include <Eigen/Core>
#include <math.h>

class Softmax {
private:
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;

public:

	static inline void softmax(const Matrix& Z, Matrix& activation) {
		activation = Z;

		for (int i = 0; i < Z.rows(); i++) {
			double maxVal = Z.row(i).maxCoeff();

			activation.row(i) = (Z.row(i).array() - maxVal).exp();
			activation.row(i) /= activation.row(i).sum();
		}
	}

	static inline void softmax_derivatives(const Matrix& output, const Matrix& true_output, Matrix& G) {
		G = output - true_output;
	}
};
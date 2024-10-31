#pragma once
#include <Eigen/Core>
#include "Output.h"

class MSE : public Output {
private:	
	Matrix d_in;

public:
	void evaluate(const Matrix& output, const Matrix& target) {
		const int rows = output.rows(), cols = output.cols();

		d_in.resize(rows, cols);
		d_in.noalias() = output - target;
	}

	const Matrix& back_prop_data() {
		return d_in;
	}

	double loss() const {
		return d_in.squaredNorm() / d_in.cols() * 0.5;
	}
};
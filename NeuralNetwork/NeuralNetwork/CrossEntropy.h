#pragma once
#include <Eigen/Core>
#include "Output.h"

class CrossEntropy : public Output {
private:
	Matrix d_in;

public:
	void evaluate(const Matrix& output, const Matrix& target) {
		const int rows = output.rows(), cols = output.cols();
		d_in.resize(rows, cols);

		d_in.noalias() = -target.cwiseQuotient(output);
	}

	void evaluate(const Matrix& output, const IntVector& target) {
		const int rows = output.rows(), cols = output.cols();
		d_in.resize(rows, cols);

		d_in.setZero();
		for (int i = 0; i < cols; i++) {
			d_in(target[i], i) = -1.0 / (output(target[i], i));
		}
	}

	const Matrix& back_prop_data() const {
		return d_in;
	}

	double loss() const {
		int n = d_in.size();
		double res = 0.0;
		const double* d_in_data = d_in.data();

		for (int i = 0; i < n; i++) {
			if (d_in_data[i] > 0) {
				res += std::log(-d_in_data[i]);
			}
		}

		return res / d_in.cols();
	}
};
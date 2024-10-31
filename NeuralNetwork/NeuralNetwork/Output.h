#pragma once
#include <Eigen/Core>

class Output {
protected:
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
	typedef Eigen::Matrix<int, Eigen::Dynamic, 1> IntVector;

public:
	~Output(){}

	virtual void evaluate(const Matrix& output, const Matrix& target) = 0;
	virtual void evaluate(const Matrix& output, const IntVector& target) = 0;
	virtual const Matrix& back_prop_data() const = 0;
	virtual double loss() const = 0;
}; 
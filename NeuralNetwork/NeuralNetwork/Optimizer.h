#pragma once
#include <Eigen/Core>

class Optimizer {
protected:
	typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
	typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
	typedef Vector::AlignedMapType AlignedMapVec;

public:
	virtual void reset() {};
	virtual void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) = 0;
};
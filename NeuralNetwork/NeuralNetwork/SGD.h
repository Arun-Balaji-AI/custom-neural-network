#pragma once
#include <Eigen/Core>
#include "Optimizer.h"

class SGD : public Optimizer {
public:
	double learning_rate, decay_rate;
	SGD() : learning_rate(0.001), decay_rate(0.0) {}

	void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) {
		vec.noalias() -= learning_rate * (dvec + decay_rate * vec);
	}

};
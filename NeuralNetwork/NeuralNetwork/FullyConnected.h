#pragma once

/*
	- This class implements the functionalities of Fully Connected Layer.
	- This inherits layers and implements the abstract functions in the layers class.
*/
#include <Eigen/Core>
#include "Utils.h"
#include <vector>
#include <stdexcept>
#include "Layer.h"
#include "softmax.h"
#include "relu.h"
#include "sigmoid.h"
#include "leakyrelu.h"
#include "linear.h"
#include "tanh.h"
#include "SGD.h"

class FullyConnected : public Layer{
private:
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
	typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
	typedef Vector::AlignedMapType AlignedMapVec;
	Matrix weights, d_weights, z, activation, d_z, d_in;
	Vector bias, d_bias;
	const std::string activation_string;

public:

	FullyConnected(const int _input_size, const int _output_size, const std::string activation) : Layer(_input_size, _output_size), activation_string(activation){
		weights.resize(_input_size, _output_size);
		d_weights.resize(_input_size, _output_size);
		bias.resize(_output_size);
		d_bias.resize(_output_size);
	}


	void intializer() {
		for (int i = 0; i < weights.rows(); i++) {
			for (int j = 0; j < weights.cols(); j++) {
				weights(i, j) = Utility::get_random_number();
			}
		}

		for (int i = 0; i < bias.rows(); i++) {
			bias[i] = Utility::get_random_number();
		}
	}

	void forward_prop(const Matrix& prev_layer_weights) override{
		const int nodes = prev_layer_weights.cols();

		// Z = (W.T) . X + B

		z.resize(this->output_size, nodes);
		z.noalias() = weights.transpose() * prev_layer_weights;
		z.colwise() += bias;

		activation.resize(this->input_size, nodes);
	}

	const Matrix& forward_prop_outputs() const override {
		return activation;
	}

	void back_prop(const Matrix& curr_layer_weights, const Matrix& next_layer_weights) override{
		Matrix delta = (next_layer_weights.transpose() * d_in).array();
		Matrix gradient = get_derivatives(activation_string);
		

		delta *= gradient;

		d_in.noalias() = curr_layer_weights * delta;
	}

	const Matrix& back_prop_outputs() const override{
		return d_in;
	}

	void update_params(Optimizer& optimizer) {
		ConstAlignedMapVec dweights(d_weights.data(), d_weights.size());
		ConstAlignedMapVec dbias(d_bias.data(), d_bias.size());
		AlignedMapVec _weights(weights.data(), weights.size());
		AlignedMapVec _bias(bias.data(), bias.size());

		optimizer.update(dweights, _weights);
		optimizer.update(dbias, _bias);
	}

	const Matrix& get_params() const override{
		return weights;
	}

	void set_params(std::vector<double>& params) {

	}

	const Matrix& get_derivatives(const std::string activation_string) const override{
		Matrix gradient;
		if (activation_string == "relu") {
			ReLu::relu_derivative(activation, gradient);
		}
		else if (activation_string == "sigmoid") {
			Sigmoid::sigmoid_derivative(activation, gradient);
		}
		else if (activation_string == "softmax") {
			// To implement...
		}
		else if (activation_string == "linear") {
			Linear::linear_derivative(activation, gradient);
		}
		else if (activation_string == "tanh") {
			Tanh::tan_h_derivative(activation, gradient);
		}
		else if (activation_string == "leakyrelu") {
			LeakyReLu::leakyrelu_derivative(activation, gradient);
		}

		return gradient;
	}
};




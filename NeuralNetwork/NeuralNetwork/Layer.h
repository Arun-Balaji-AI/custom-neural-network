#pragma once
/*
	- This class consists of the functions that are needed when a layer is defined.
	- Layer class is inherited by other classes like convolution, fully connected, etcetra. So the functions are mostly pure vitual functions.
	- Each Layer uses Eigen Matrices and Vectors instead of regular vectors. This is primarily due to the flexibility to linear algebra functions by Eigens.
	- This class contains,
		* get_input_size()
		* get_output_size()
		* initializer()
		* forward_prop()
		* back_prop()
		* forward_prop_outputs()
		* back_prop_outputs()
		* get_params()
		* set_params()
*/
#include<Eigen/Core>
#include <random>
#include <vector>

class Layer {
protected:
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;

	const int input_size, output_size;

public:
	Layer(int _input_size, int _output_size) : input_size(_input_size), output_size(_output_size){}

	int get_input_size() {
		return input_size;
	}

	int get_output_size() {
		return output_size;
	}

	virtual void initializer() = 0;
	
	virtual void forward_prop(const Matrix& prev_layer_weights) = 0;

	virtual void back_prop(const Matrix& curr_layer_weights, const Matrix& next_layer_weights) = 0;

	virtual const Matrix& forward_prop_outputs() const = 0;

	virtual const Matrix& back_prop_outputs() const = 0;

	virtual const Matrix& get_params() const {};

	virtual void set_params(const std::vector<double>& params) = 0;

	virtual const Matrix& get_derivatives(const std::string activation_string) const = 0;
	~Layer(){}
};
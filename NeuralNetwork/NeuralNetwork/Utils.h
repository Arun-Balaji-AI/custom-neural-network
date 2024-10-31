#pragma once
/*
	- This header file contains the all needed functions that is used while implementing a custom neural network
	- This contains,
		* generate_random_numbers()
*/

#include <random>

double generate_random_number() {
	/*
		- This function generates a random number using normal distribution with mean = 0.0, and standard_deviation = 1.0.
		- Parameters : None
		  Returns : A double number generated through normal distribution.
	*/
	std::random_device rd;
	std::mt19937 generator(rd());
	std::normal_distribution<double> random_number(0.0, 1.0);

	return random_number(generator);
}


namespace Utility {
	double get_random_number() {
		return generate_random_number();
	}
};
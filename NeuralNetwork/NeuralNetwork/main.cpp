#include <Eigen/Dense>
#include <iostream>

int main() {
	Eigen::MatrixXd m = Eigen::MatrixXd::Random(3, 3);

	m += (Eigen::MatrixXd::Constant(3, 3, 1.2));

	std::cout << Eigen::MatrixXd::Constant(3, 3, 1.2) * 2 << std::endl << std::endl;

	std::cout << m(4) << std::endl;
	Eigen::VectorXd v(3);
	v << 5, 89, 8;
	std::cout << std::endl << m * v << std::endl;

	return 0;
}
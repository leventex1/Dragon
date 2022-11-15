#include "UtilityFunctions.h"

DRAGON_BEGIN

Tensor1D convertTo(const Tensor& tensor, size_t cols) {
	return Tensor1D(cols, tensor.getData());
}

Tensor1D convertTo(Tensor&& tensor, size_t cols) {
	return Tensor1D(cols, std::move(tensor));
}

void print(const Tensor3D& tensor) {
	for (size_t k = 0; k < tensor.getDepth(); k++) {
		for (size_t i = 0; i < tensor.getRows(); i++) {
			for (size_t j = 0; j < tensor.getCols(); j++) {
				std::cout << tensor.at(i, j, k) << "\t";
			}
			std::cout << "\n";
		}
		std::cout << "\n\n";
	}
	std::cout << "\n";
}

void print(const Tensor2D& tensor) {
	for (size_t i = 0; i < tensor.getRows(); i++) {
		for (size_t j = 0; j < tensor.getCols(); j++) {
			std::cout << tensor.at(i, j) << "\t";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

void print(const Tensor& tensor) {
	for (size_t i = 0; i < tensor.getCount(); i++)
		std::cout << tensor.getData()[i] << " ";
	std::cout << "\n\n";
}

Tensor2D trans(const Tensor2D& tensor) {
	Tensor2D result(tensor.getCols(), tensor.getRows(), precision());
	for (size_t i = 0; i < result.getRows(); i++)
		for (size_t j = 0; j < result.getCols(); j++)
			result.at(i, j) = tensor.at(j, i);
	return result;
}

Tensor2D trans(const Tensor1D& tensor) {
	return Tensor2D(1, tensor.getCols(), tensor);
}

Tensor2D optrans(const Tensor2D& tensor) {
	Tensor2D result(tensor.getCols(), tensor.getRows(), precision());
	for (size_t i = 0; i < result.getRows(); i++)
		for (size_t j = 0; j < result.getCols(); j++)
			result.at(i, j) = tensor.at(tensor.getRows() - j - 1, tensor.getCols() - i - 1);
	return result;
}

Tensor2D reverse(const Tensor2D& tensor) {
	precision* assingPointer = new precision[tensor.getRows() * tensor.getCols()];

	for (size_t i = 0; i < tensor.getRows(); i++)
		for (size_t j = 0; j < tensor.getCols(); j++)
			assingPointer[(tensor.getRows() - i - 1) * tensor.getCols() + tensor.getCols() - j - 1] = tensor.at(i, j);

	return Tensor2D(assingPointer, tensor.getRows(), tensor.getCols());
}

Tensor2D tensorDot(const Tensor2D& left, const Tensor2D& right) {
	Tensor2D result(left.getRows(), right.getCols(), precision());
	assert((left.getCols() == right.getRows()) && "Parameters not match for tensorDot!");

		for (size_t i = 0; i < result.getRows(); i++) {
			for (size_t j = 0; j < result.getCols(); j++) {

				precision prod = precision();
				for (size_t t = 0; t < left.getCols(); t++) {
					prod += left.at(i, t) * right.at(t, j);
				}

				result.at(i, j) = prod;

			}
		}

	return result;
}

Tensor2D tensorDot(const Tensor2D& left, const Tensor1D& right) {
	Tensor2D result(left.getRows(), 1, precision());
	assert((left.getCols() == right.getCols()) && "Parameters not match for tensorDot!");

	for (size_t i = 0; i < result.getRows(); i++) {

		precision prod = precision();
		for (size_t t = 0; t < left.getCols(); t++)
			prod += left.at(i, t) * right.at(t);


		result.at(i, 0) = prod;
	}
	
	return result;
}

Tensor2D tensorDot(const Tensor1D& left, const Tensor2D& right) {
	Tensor2D result(left.getCols(), right.getCols(), precision());
	assert((right.getRows() == 1) && "Parameters not match for tensorDot!");

	for (size_t i = 0; i < result.getRows(); i++)
		for (size_t j = 0; j < result.getCols(); j++)
			result.at(i, j) = left.at(i) * right.at(0, j);

	return result;
}

precision tensorDot(const Tensor1D& left, const Tensor1D& right) {
	assert((right.getCols() == left.getCols()) && "Parameters not match for tensorDot! Two vector has different sizez.");
	precision prod = precision();
	for (size_t i = 0; i < right.getCols(); i++)
		prod += right.at(i) * left.at(i);
	return prod;
}

Tensor2D padding(const Tensor2D& tensor, size_t size, precision value) {
	Tensor2D result(tensor.getRows() + 2 * size, tensor.getCols() + 2 * size, value);

	for (size_t i = 0; i < tensor.getRows(); i++)
		for (size_t j = 0; j < tensor.getCols(); j++)
			result.at(size + i, size + j) = tensor.at(i, j);

	return result;
}

Tensor2D convolution(const Tensor2D& signal, const Tensor2D& kernel, size_t stride) {
	size_t r = size_t((signal.getRows() - kernel.getRows()) / stride) + 1;
	size_t c = size_t((signal.getCols() - kernel.getCols()) / stride) + 1;

	double* assignPointer = new double[r * c];

	for (size_t i = 0; i < r; i++) {
		for (size_t j = 0; j < c; j++) {

			size_t sr = i * stride;
			size_t sc = j * stride;

			double product = 0.0;
			for (size_t x = 0; x < kernel.getRows(); x++)
				for (size_t y = 0; y < kernel.getCols(); y++)
					product += signal.at(sr + x, sc + y) * kernel.at(x, y);

			assignPointer[i * c + j] = product;

		}
	}
	return Tensor2D(assignPointer, r, c);
}

void convolution(Tensor2D& result, const Tensor2D& signal, const Tensor2D& kernel, size_t stride) {
	size_t r = size_t((signal.getRows() - kernel.getRows()) / stride) + 1;
	size_t c = size_t((signal.getCols() - kernel.getCols()) / stride) + 1;
	assert(r == result.getRows() && c == result.getCols());


	for (size_t i = 0; i < r; i++) {
		for (size_t j = 0; j < c; j++) {

			size_t sr = i * stride;
			size_t sc = j * stride;

			double product = 0.0;
			for (size_t x = 0; x < kernel.getRows(); x++)
				for (size_t y = 0; y < kernel.getCols(); y++)
					product += signal.at(sr + x, sc + y) * kernel.at(x, y);

			result.at(i, j) = product;
		}
	}
}

Tensor2D scaleByStride(const Tensor2D& signal, size_t stride) {
	size_t row = (signal.getRows() - 1) * stride + 1;
	size_t col = (signal.getCols() - 1) * stride + 1;
	
	Tensor2D result(row, col, 0.0);

	for (size_t i = 0; i < signal.getRows(); i++)
		for (size_t j = 0; j < signal.getCols(); j++)
			result.at(i * stride, j * stride) = signal.at(i, j);

	return result;
}

size_t calcConvParamsAfter(size_t inputPar, size_t kernelPar, size_t stride) {
	return size_t((inputPar - kernelPar) / stride) + 1;
}

DRAGON_END
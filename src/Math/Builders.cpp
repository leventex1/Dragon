#include "Builders.h"

DRAGON_BEGIN

Tensor2D unit(size_t N) {
	Tensor2D result(N, N, 0.0);
	for (size_t i = 0; i < N; i++)
		result.at(i, i) = (precision)(1.0);
	return result;
}

Tensor2D random(size_t rows, size_t cols, precision min, precision max) {
	std::random_device rg;

	Tensor2D result(rows, cols, min);
	for (size_t i = 0; i < rows; i++)
		for (size_t j = 0; j < cols; j++)
			result.at(i, j) = min + ((precision)rg() / (precision)rg.max()) * (max - min);
	return result;
}

Tensor1D random(size_t count, precision min, precision max) {
	std::random_device rg;

	precision* assignData = new precision[count];
	for (size_t i = 0; i < count; i++)
		assignData[i] = min + ((precision)rg() / (precision)rg.max()) * (max - min);

	return Tensor1D(assignData, count);
}

Tensor2D randomInt(size_t rows, size_t cols, precision min, precision max) {
	std::random_device rg;

	Tensor2D result(rows, cols, min);
	for (size_t i = 0; i < rows; i++)
		for (size_t j = 0; j < cols; j++)
			result.at(i, j) = floor(min + ((precision)rg() / (precision)rg.max()) * (max - min));
	return result;
}

Tensor1D randomInt(size_t count, precision min, precision max) {
	std::random_device rg;

	precision* assignData = new precision[count];
	for (size_t i = 0; i < count; i++)
		assignData[i] = floor(min + ((precision)rg() / (precision)rg.max()) * (max - min));

	return Tensor1D(assignData, count);
}

Tensor2D randomD(size_t rows, size_t cols, precision mean, precision dev) {
	std::default_random_engine generator;
	std::normal_distribution<precision> distribution(mean, dev);
	Tensor2D result(rows, cols, precision());
	for (size_t i = 0; i < result.getCount(); i++)
		result.getData()[i] = distribution(generator);
	return result;
}

Tensor1D randomD(size_t count, precision mean, precision dev) {
	std::default_random_engine generator;
	std::normal_distribution<precision> distribution(mean, dev);

	precision* assignData = new precision[count];
	for (size_t i = 0; i < count; i++)
		assignData[i] = distribution(generator);

	return Tensor1D(assignData, count);
}

Tensor1D initTensor(size_t count, std::function<precision()> initFunction) {
	precision* assignData = new precision[count];

	for (size_t i = 0; i < count; i++)
		assignData[i] = initFunction();

	return Tensor1D(assignData, count);
}

DRAGON_END
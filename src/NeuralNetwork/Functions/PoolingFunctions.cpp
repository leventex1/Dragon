#include "PoolingFunctions.h"

DRAGON_BEGIN

double maxPool(const std::vector<double>& values) {
	size_t index = 0;
	for (size_t i = 1; i < values.size(); i++)
		if (values[i] > values[index])
			index = i;
	return values[index];
}

std::vector<double> maxPoolDiff(const std::vector<double>& values, double value) {
	std::vector<double> result(values.size(), 0.0);
	size_t index = 0;
	for (size_t i = 1; i < values.size(); i++)
		if (values[i] > values[index])
			index = i;
	result[index] = value;
	return result;
}

DRAGON_END
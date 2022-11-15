#pragma once
#include "../Core.h"

#include <random>
#include <math.h>
#include <functional>

#include "Tensor.h"
#include "Tensor1D.h"
#include "Tensor2D.h"

DRAGON_BEGIN

// Create a unit tensor(matrix)
DRAGON_API Tensor2D unit(size_t N);
// Create a tensor with random values
DRAGON_API Tensor2D random(size_t rows, size_t cols, precision min, precision max);
DRAGON_API Tensor1D random(size_t count, precision min, precision max);
// Create a tensor with random values converted to int
DRAGON_API Tensor2D randomInt(size_t rows, size_t cols, precision min, precision max);
DRAGON_API Tensor1D randomInt(size_t count, precision min, precision max);
// Create a tensor with gaussian distribution
// mean: mean of the distribution, dev: deviation of the distribution
DRAGON_API Tensor2D randomD(size_t rows, size_t cols, precision mean, precision dev);
DRAGON_API Tensor1D randomD(size_t count, precision mean, precision dev);

// Create a Tensor with every element assign to the output of the InitFunction
DRAGON_API Tensor1D initTensor(size_t count, std::function<precision()> initFunction);

// Create a tensor type by adding them together elementwise.
template<class TensorType>
TensorType add(const TensorType& A, const TensorType& B) {
	TensorType result(A);
	result.add(B);
	return result;
}

// Create a tensor type by subtracting B from A elementwise.
template<class TensorType>
TensorType sub(const TensorType& A, const TensorType& B) {
	TensorType result(A);
	result.sub(B);
	return result;
}

// Create a tensor type by multiplying them together elementwise.
template<class TensorType>
TensorType mult(const TensorType& A, const TensorType& B) {
	TensorType result(A);
	result.mult(B);
	return result;
}

// Create a tensor type by dividing A by B elementwise.
template<class TensorType>
TensorType div(const TensorType& A, const TensorType& B) {
	TensorType result(A);
	result.div(B);
	return result;
}

DRAGON_END
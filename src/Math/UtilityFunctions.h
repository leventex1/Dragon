#pragma once
#include "../Core.h"
#include <iostream>

#include "Tensor.h"
#include "Tensor1D.h"
#include "Tensor2D.h"
#include "Tensor3D.h"

DRAGON_BEGIN

// Converts tensors to each other
DRAGON_API Tensor1D convertTo(const Tensor& tensor, size_t cols);
DRAGON_API Tensor1D convertTo(Tensor&& tensor, size_t cols);

// Prints The 2D tensor to the consol
DRAGON_API void print(const Tensor3D& tensor);
DRAGON_API void print(const Tensor2D& tensor);
DRAGON_API void print(const Tensor& tensor);
//void print(Tensor&& tensor);
// Calculate the transponant of the 2D tensor
DRAGON_API Tensor2D trans(const Tensor2D& tensor);
DRAGON_API Tensor2D trans(const Tensor1D& tensor);
// Calulate the opponent transponant of the 2D tensor
// Rotating the tensor at the axis of the 2nd axis
DRAGON_API Tensor2D optrans(const Tensor2D& tensor);
// Create a tnesor with rows and cols swaped at the middle point
// Or calculate the trans of the tensor and than the optrans of the resulted tensor.
DRAGON_API Tensor2D reverse(const Tensor2D& tensor);
// Calculate the Matrix multiplication between two tensor
DRAGON_API Tensor2D tensorDot(const Tensor2D& left, const Tensor2D& right);
// Calculate the Matrix multiplication between two tensor the right tensor handled as a column vector
DRAGON_API Tensor2D tensorDot(const Tensor2D& left, const Tensor1D& right);
// Calculate the Matrix multiplication between two tensor the left tensor handled as a column vector
DRAGON_API Tensor2D tensorDot(const Tensor1D& left, const Tensor2D& right);
// Calucuate the matematical scalar vector prouduct
DRAGON_API precision tensorDot(const Tensor1D& left, const Tensor1D& right);
// Creating a tensor but adds extra elements to the side
DRAGON_API Tensor2D padding(const Tensor2D& tensor, size_t size, precision value);
// Calculate the convolution between two tensor.
// The Kernel tensor goes through the single tensor with stirde steps and calculate the 
// product between the subsignal with size of kernel and the kenrnel.
DRAGON_API Tensor2D convolution(const Tensor2D& signal, const Tensor2D& kernel, size_t stride);
//Tensor2D convolution(Tensor2D&& signal, const Tensor2D& kernel, size_t stride).
DRAGON_API void convolution(Tensor2D& result, const Tensor2D& signal, const Tensor2D& kernel, size_t stride);
// Scale the tensor by stride (function needed for convolutional layer backprop).
DRAGON_API Tensor2D scaleByStride(const Tensor2D& signal, size_t stride);

// calculate the resulted parameter after a convolutional operation occur on the input by the kernel
// inputPar = inputRow,Col... kernelPar = kernelRow, Col...
DRAGON_API size_t calcConvParamsAfter(size_t inputPar, size_t kernelPar, size_t stride);

DRAGON_END
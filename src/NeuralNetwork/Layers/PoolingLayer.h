#pragma once

#include <vector>

#include "BaseLayer.h"

DRAGON_BEGIN

/// <summary>
/// Pooling layer is the child of Base layer.
/// This layer taks in a Tensor and down sample it with a pooling function.
/// This layer usully comes after a convolutional layer.
/// </summary>
class DRAGON_API PoolingLayer : public BaseLayer {
public:

	PoolingLayer();
	PoolingLayer(
		size_t inputRows, size_t inputCols, size_t inputDepth,
		size_t kernelRows, size_t kernelCols,
		const std::function<double(const std::vector<double>& values)> poolingFunc,
		const std::function<std::vector<double>(const std::vector<double>& values, double value)> poolingFuncDiff);
	PoolingLayer(const PoolingLayer& other);
	PoolingLayer(PoolingLayer&& other) noexcept;
	
	Tensor1D feedForward(const Tensor1D& input) const override;

	Tensor1D backPropagate(
		Tensor1D& sumsAfter,
		Tensor1D& costAfter,
		Tensor1D& activationsBefore,
		double learningRate) override;

	PreparePropagateData preparePropagate(const Tensor1D& input) const override;

	std::string toString() const override;
	void fromString(const std::string& rawString) override;
	std::string getName() const override { return "PoolingLayer"; }

private:
	// inputType[0] = inputRows, inputType[1] = inputCols, inputType[2] = inputDepth
	ParameterType<3> m_InputType;
	size_t m_KernelRows = 0;
	size_t m_KernelCols = 0;
	std::function<double(const std::vector<double>& values)> m_PoolingFunction;
	std::function<std::vector<double>(const std::vector<double>& values, double value)> m_PoolingFunctionDiff;
};

DRAGON_END
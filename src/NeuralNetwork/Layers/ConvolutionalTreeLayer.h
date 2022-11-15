#pragma once

#include "BaseLayer.h"

DRAGON_BEGIN

class DRAGON_API ConvolutionalTreeLayer : public BaseLayer {
public:
	ConvolutionalTreeLayer();
	ConvolutionalTreeLayer(
		size_t inputRows, size_t inputCols, size_t inputDepth,
		size_t kernelRows, size_t kernelCols, size_t kernelCountPerInput,
		size_t kernelStride,
		const std::function<double()>& initFunction,
		const ActivationFunction& activation);
	ConvolutionalTreeLayer(const ConvolutionalTreeLayer& other);
	ConvolutionalTreeLayer(ConvolutionalTreeLayer&& other) noexcept;


	inline const Tensor3D& getKernels() const { return m_Kernels; }
	inline Tensor3D& getKernels() { return m_Kernels; }
	inline const Tensor3D& getBiases() const { return m_Biases; }
	inline Tensor3D& getBiases() { return m_Biases; }
	inline const size_t& getKernelStride() const { return m_KernelStride; }
	inline size_t& getKernelStride() { return m_KernelStride; }

public:

	Tensor1D feedForward(const Tensor1D& input) const override;

	Tensor1D backPropagate(
		Tensor1D& sumsAfter,
		Tensor1D& costAfter,
		Tensor1D& activationsBefore,
		double learningRate) override;

	PreparePropagateData preparePropagate(const Tensor1D& input) const override;

	std::string toString() const override;
	void fromString(const std::string& rawString) override;
	std::string getName() const override { return "ConvolutionalTreeLayer"; }

private:
	// [0] = input rows, [1] = input cols, [2] = input depth
	ParameterType<3> m_InputType;
	// [0] = output rows, [1] = output cols [2] = output depth = inputDept * kernelCountPerInput
	ParameterType<3> m_OutputType;

	size_t m_KernelStride;
	Tensor3D m_Kernels;
	Tensor3D m_Biases;
};

DRAGON_END
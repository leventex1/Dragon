#pragma once

#include "BaseLayer.h"

DRAGON_BEGIN

/// <summary>
/// Dense Layer is the child of Base Layer.
/// Basic fully conected layer.
/// Member variables are Weight tensor, Biases tensor and the InputType stucture.
/// InputType contains the number of input nodes and the number of output nodes.
/// </summary>
class DRAGON_API DenseLayer : public BaseLayer {
private:
	// m_InputType.parameters[0] = numInputNodes, m_inputType.parameters[1] = numOutputNodes
	ParameterType<2> m_InputType;

	Tensor2D m_Weights;
	Tensor1D m_Biases;
public:
	DenseLayer();
	DenseLayer(
		size_t numInputNodes, size_t numOutputNodes,
		const std::function<double()>& initFunction,
		const ActivationFunction& activation);
	DenseLayer(const DenseLayer& other);
	DenseLayer(DenseLayer&& other) noexcept;


	inline const Tensor2D& getWeights() const { return m_Weights; }
	inline const Tensor1D& getBiases() const { return m_Biases; }
	inline Tensor2D& getWeights() { return m_Weights; }
	inline Tensor1D& getBiases() { return m_Biases; }

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
	std::string getName() const override { return "DenseLayer"; }
};

DRAGON_END
#include "DenseLayer.h"

DRAGON_BEGIN

DenseLayer::DenseLayer() : m_InputType({ 0, 0 }) { }

DenseLayer::DenseLayer(
	size_t numInputNodes, size_t numOutputNodes,
	const std::function<double()>& initFunction,
	const ActivationFunction& activation) :
	m_InputType({ numInputNodes, numOutputNodes }), 
	BaseLayer(activation) {

	m_Weights = Tensor2D(numOutputNodes, numInputNodes, initTensor(m_InputType.getParameterCount(), initFunction));
	m_Biases = initTensor(numOutputNodes, initFunction);
}

DenseLayer::DenseLayer(const DenseLayer& other) :
	m_InputType(other.m_InputType),
	m_Weights(other.m_Weights),
	m_Biases(other.m_Biases),
	BaseLayer(other.m_Activation) { }

DenseLayer::DenseLayer(DenseLayer&& other) noexcept :
	m_InputType(other.m_InputType),
	m_Weights(std::move(other.m_Weights)),
	m_Biases(std::move(other.m_Biases)),
	BaseLayer(other.m_Activation) { }

Tensor1D DenseLayer::feedForward(const Tensor1D& input) const {
	assert((input.getCount() == m_InputType.parameters[0]) && "Invalid input parameters!");

	Tensor2D working = tensorDot(m_Weights, input);
	working.add(m_Biases);
	working.manipul(m_Activation.getActivation());
	return Tensor1D(m_InputType.parameters[1], std::move(working));
}

Tensor1D DenseLayer::backPropagate(
	Tensor1D& sumsAfter,
	Tensor1D& costAfter, 
	Tensor1D& activationBefore, 
	double learningRate) {

	assert((activationBefore.getCount() == m_InputType.parameters[0]) && "Invalid before activation parameters!");
	assert((sumsAfter.getCount() == costAfter.getCount() && sumsAfter.getCount() == m_InputType.parameters[1]) &&
		"Invalid cost and sums parameters!");

	sumsAfter.manipul(m_Activation.getActivationDiff()).mult(costAfter);
	Tensor2D gradientWeight = tensorDot(sumsAfter, trans(activationBefore));
	Tensor1D gradientBiases = sumsAfter;
	Tensor1D costBefore = Tensor1D(m_InputType.parameters[0], 
		tensorDot(trans(m_Weights), Tensor2D(m_InputType.parameters[1], 1, std::move(sumsAfter))));

	m_Weights.sub(gradientWeight.mult(learningRate));
	m_Biases.sub(gradientBiases.mult(learningRate));
	return costBefore;
}

PreparePropagateData DenseLayer::preparePropagate(const Tensor1D& input) const {
	assert((input.getCount() == m_InputType.parameters[0]) && "Invalid input parameters!");
	PreparePropagateData pData;
	
	Tensor2D working = tensorDot(m_Weights, input);
	working.add(m_Biases);

	pData.input = Tensor1D(input);
	pData.sum = Tensor1D(m_InputType.parameters[1], working);
	pData.output = Tensor1D(m_InputType.parameters[1], std::move(working.manipul(m_Activation.getActivation())));

	return pData;
}

std::string DenseLayer::toString() const {
	// Create a string stream. 
	std::stringstream ss;
	ss << std::fixed << std::setprecision(8);

	// Put the dense layer parameter dimensions.
	// Input nodes.
	ss << std::to_string(m_InputType.parameters[0]) << " ";
	// Output nodes.
	ss << std::to_string(m_InputType.parameters[1]) << " ";

	// Put the weights.
	for (size_t i = 0; i < m_Weights.getCount(); i++) {
		ss << m_Weights.getData()[i] << " ";
	}
	// Put the biases.
	for (size_t i = 0; i < m_Biases.getCount(); i++) {
		ss << m_Biases.getData()[i] << " ";
	}

	return ss.str();
}

void DenseLayer::fromString(const std::string& rawString) {
	std::stringstream ss(rawString);

	// Get the input nodes.
	ss >> m_InputType.parameters[0];
	// Get the output nodes.
	ss >> m_InputType.parameters[1];

	// Create the weight and biases matrces.
	m_Weights = Tensor2D(m_InputType.parameters[1], m_InputType.parameters[0], 0.0);
	m_Biases = Tensor1D(m_InputType.parameters[1], 0.0);
	// Get the weights.
	for (size_t i = 0; i < m_Weights.getCount(); i++) {
		ss >> m_Weights.getData()[i];
	}
	// Get the biases.
	for (size_t i = 0; i < m_Biases.getCount(); i++) {
		ss >> m_Biases.getData()[i];
	}
}

DRAGON_END
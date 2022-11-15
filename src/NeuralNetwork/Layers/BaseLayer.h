#pragma once
#include <functional>
#include <array>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>

#include "../../Core.h"

#include "../../Math/MathCore.h"
#include "../Functions/Functions.h"

DRAGON_BEGIN


// Holds some usefull parameters in the layers.
template<size_t NumParams>
struct ParameterType {
	std::vector<size_t> parameters;

	ParameterType(std::initializer_list<size_t> initList)
		: parameters(initList) { }

	size_t getParameterCount() const {
		size_t count = 1;
		for (size_t i = 0; i < NumParams; i++)
			count *= parameters[i];
		return count;
	}
};


/// <summary>
/// PreparaPropagateData holds the data for the backPropagate algorithm.
/// For training the model firts we need to push through the training data in the model and
/// harvest every layers's inputs(input), weighted sums(sum), activations(output).
/// At the end of the mondel we calculate a cost, and propagate backward with that cost 
/// and tha saved data.
/// </summary>
struct DRAGON_API PreparePropagateData {
	Tensor1D input;
	Tensor1D sum;
	Tensor1D output;

	PreparePropagateData() = default;
	PreparePropagateData(PreparePropagateData&& other) noexcept :
		input(std::move(other.input)),
		sum(std::move(other.sum)),
		output(std::move(other.output)) { }
};


/// <summary>
/// Base Layer is the root of every layer.
/// Member variables are the two activation function.
///	m_Activation is need for the feedForward algorithm to calculate the layer activations.
/// m_ActivationDiff is need for the backPropagete algorithm to calulate the gradients.
/// </summary>
class DRAGON_API BaseLayer {
protected:
	ActivationFunction m_Activation;

public:
	BaseLayer() = default;
	BaseLayer(const ActivationFunction& activation);

	inline ActivationFunction& getActivation() { return m_Activation; }
	inline const ActivationFunction& getActivation() const { return m_Activation; }

public:

	// Flow the data forward in the layer.
	virtual Tensor1D feedForward(const Tensor1D& input) const = 0;
	
	//Flow the cost gradient backward in tha layer.
	//Takes the gradient(costAfter), and the layer concerning data like
	//the sum of the nodes and thier activations.
	//Learning rate parameter defines how quick or slow the model is approaching the target value.
	virtual Tensor1D backPropagate(
		Tensor1D& sumsAfter,
		Tensor1D& costAfter,
		Tensor1D& activationsBefore,
		double learningRate) = 0;

	// Push the data forward and save the layer calculations.
	virtual PreparePropagateData preparePropagate(const Tensor1D& input) const = 0;


	// Save the layer's parameters into a string, wich can be saved into a file.
	// The weight are rounded to 8 decimal digits.
	virtual std::string toString() const = 0;
	// Load the layer's parameters from a string, wich can be loaded from a file.
	virtual void fromString(const std::string& rawString) = 0;
	// Get the layer's name(class name) as a string.
	virtual std::string getName() const = 0;
};

DRAGON_END
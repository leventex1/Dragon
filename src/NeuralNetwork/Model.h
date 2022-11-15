#pragma once
#include <vector>
#include <functional>
#include <string>

#include "Layers/Layers.h"

DRAGON_BEGIN

/// <summary>
/// Layer stuct containes all the layer information.
/// </summary>
struct Layer {
	BaseLayer* layer;			// Contains the layer parameters
	bool isInitializedOnModel;	// True if the layer initialized on the model object, than it's need to be deleted by the model object.
};

/// <summary>
/// Model class wraps the Layers together creating a neuralnetwork.
/// Model itelf just holding the network.
/// Different functions using this model to train and do analysis.
/// To Create a NeuralNetwork create a model, and use the template addLayer method
/// to specify what kind of layer you want to add to the model.
/// [example]
/// Model model;
/// model.addLayer<DenseLayer>(DenseLayer(2, 3, initFunction, sigmoid, sigmoidDiff));
/// model.addLayer<DenseLayer>(DenseLayer(3, 1, initFunction, sigmoid, sigmoidDiff));
/// To save the model into file, use the save function with a file path and an extension of .txt.
/// To load the model from file, use the load funtcoin with a file path and an extension of .txt.
/// Note that if you have costum layer or functions, first you need to provide the model with
/// the specific createLayer and createActivation functions.
/// </summary>
class DRAGON_API Model {
public:
	Model() = default;
	~Model();
	 
	void addLayer(BaseLayer* layer);

	// Adds your costum layerCreator.
	// If you want to load your previously saved model, first you need to provide the model with your costum layerCreator function.
	// The function should take in a string(layerName like DenseLayer) and contruct a new Layer on the heap and return it as BaseLayer*.
	void addLayerCreatorFunction(const std::function<BaseLayer* (const std::string& layerName)>& creator);
	// Adds your costum activationFunction creator.
	// If you want to load your previously saved model, first you need to provide the model with your costum activationfunction creator function.
	// The function should thake in a string(activation name like sigmoid) and construct an ActivationFunction object and return it.
	void addActivationCreatorFunction(const std::function<ActivationFunction(const std::string& activationName)>& creator);

	// Save your model to file (txt).
	void save(const std::string& filePath);
	// Load your model from file (txt).
	void load(const std::string& filePath);

	inline const std::vector<Layer>& getLayers() const { return m_Layers; }
	inline std::vector<Layer>& getLayers() { return m_Layers; }

private:
	// Create a Layer in the load function from the given layerName.
	BaseLayer* createLayer(const std::string& layerName);
	// Create an Activation in the load function from the given layerName.
	ActivationFunction createActivation(const std::string& activationName);

private:
	// This member holds the costum layer creator functions to load costum layers.
	// If you create costum layer and want to load it to your model, first create a layerCreator function,
	// It should take a string(layer name) as parameter and returns a heap allocated Layer pointer as Baselayer pointer.
	// The load and save functions will use your layer's getName function and give it to the layerCreator as a parameter.
	// The layerConstructor than gives back a BaseLayer pointer if the layerName and the name from the file matches.
	std::vector<std::function<BaseLayer* (const std::string& layerName)>> m_LayerCreator;
	std::vector<std::function<ActivationFunction(const std::string& functionName)>> m_ActivationCreator;
	// This member holds the layers as BaseLayer pointer.
	std::vector<Layer> m_Layers;
};

// Push the input data through the model and returns the output.
DRAGON_API Tensor1D feedForward(const Model& neuralNetwork, const Tensor1D& input);

// Superwised learning (Classification)
// CostFunction parameter should return the cost differentiated by the output.
// Returns the differentiated cost respect to the input
DRAGON_API Tensor1D trainModel(
	Model& neuralNetwork,
	const Tensor1D& input,
	const Tensor1D& target,
	const std::function<Tensor1D(const Tensor1D& output, const Tensor1D& target)>& costFunction,
	double learningRate);

DRAGON_END
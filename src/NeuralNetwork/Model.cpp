#include "Model.h"

DRAGON_BEGIN

Model::~Model() { 
	for (size_t i = 0; i < m_Layers.size(); i++) {
		if (m_Layers[i].isInitializedOnModel) {
			delete m_Layers[i].layer;
		}
	}
}

void Model::addLayer(BaseLayer* layer) {
	m_Layers.push_back({ layer, false });
}

void Model::addLayerCreatorFunction(const std::function<BaseLayer* (const std::string& layerName)>& creator) {
	m_LayerCreator.push_back(creator);
}

void Model::addActivationCreatorFunction(const std::function<ActivationFunction(const std::string& activationName)>& creator) {
	m_ActivationCreator.push_back(creator);
}

BaseLayer* Model::createLayer(const std::string& layerName) {
	BaseLayer* newLayer = nullptr;

	// Built in layers.
	DenseLayer dummyDense;
	ConvolutionalLayer dummyConv;
	ConvolutionalTreeLayer dummyConvTree;
	PoolingLayer dummyPool;

	if (layerName == dummyDense.getName())
		newLayer = new DenseLayer();
	else if (layerName == dummyConv.getName())
		newLayer = new ConvolutionalLayer();
	else if (layerName == dummyConvTree.getName())
		newLayer = new ConvolutionalTreeLayer();
	else if (layerName == dummyPool.getName())
		newLayer = new PoolingLayer();

	// Costum created layers.
	if (!newLayer) {
		for (size_t i = 0; i < m_LayerCreator.size(); i++) {
			BaseLayer* temp = m_LayerCreator[i](layerName);
			if (temp) {
				newLayer = temp;
				break;
			}
		}
	}

	return newLayer;
}

ActivationFunction Model::createActivation(const std::string& activationName) {
	ActivationFunction activation;

	// Built in activation functions.
	if (activationName.compare(sigmoid().getName()) == 0) {
		activation = sigmoid();
	}
	else if (activationName == relU().getName()) {
		activation = relU();
	} 
	else if (activationName == relU10().getName()) {
		activation = relU10();
	}
	else if (activationName == relU100().getName()) {
		activation = relU100();
	}
	else if (activationName == relU500().getName()) {
		activation = relU500();
	}

	if (activation.getName().empty()) {
		for (size_t i = 0; i < m_ActivationCreator.size(); i++) {
			ActivationFunction temp = m_ActivationCreator[i](activationName);
			if (!temp.getName().empty()) {
				activation = temp;
				break;
			}
		}
	}

	return activation;
}

void Model::save(const std::string& filePath) {
	
	std::string outData = std::to_string(m_Layers.size()) + "!\n";
	for (size_t i = 0; i < m_Layers.size(); i++) {
		outData += 
			m_Layers[i].layer->getName() + "; " +					// Put layer name
			m_Layers[i].layer->getActivation().getName() + "@ " +	// Put layer activatoin
			m_Layers[i].layer->toString() + "!\n";					// Put layer data
	}

	std::ofstream file(filePath);
	if (file.is_open())
		file << outData;
	else
		std::cout << "Couldn't open file at: " << filePath << std::endl;
}

void Model::load(const std::string& filePath) {

	std::ifstream file(filePath);
	if (!file.is_open()) {
		std::cout << "Couldn't open file at: " << filePath << std::endl;
		return;
	}

	std::stringstream ss;
	std::string inData = "";
	ss << file.rdbuf();

	size_t numLayers = 0;
	ss >> numLayers;

	inData = ss.str();
	inData = inData.substr(inData.find('!') + 1);

	for (size_t i = 0; i < numLayers; i++) {
		std::string layerName = inData.substr(1, inData.find(';') - 1);
		std::string layerActivation = inData.substr(layerName.size() + 3, inData.find('@') - layerName.size() - 3);
		std::string layerData = inData.substr(layerName.size() + 2 + layerActivation.size() + 2, inData.find('!') - layerName.size() - 1);

		BaseLayer* newLayer = createLayer(layerName);
		ActivationFunction activation = createActivation(layerActivation);

		if (newLayer) {
			newLayer->fromString(layerData);
			m_Layers.push_back({ newLayer, true });
		}
		else {
			std::cout << "Couldn't construct layer! Please provide a layerCreator to the model with a name of " << 
				layerName << "!" << std::endl;
		}

		if (!activation.getName().empty()) {
			newLayer->getActivation() = activation;
		}
		else {
			std::cout << "Couldn't construct activation! Please provide an activationCreator to the model with a name of " <<
				layerActivation << "!" << std::endl;
		}

		inData = inData.substr(inData.find('!') + 1);
	}
		
}


Tensor1D feedForward(const Model& neuralNetwork, const Tensor1D& input) {
	Tensor1D working = input;
	const std::vector<Layer>& layers = neuralNetwork.getLayers();

	// Push the data through the layers.
	for (size_t i = 0; i < layers.size(); i++)
		working = layers[i].layer->feedForward(working);

	return working;
}

Tensor1D trainModel(
	Model& neuralNetwork,
	const Tensor1D& input,
	const Tensor1D& target,
	const std::function<Tensor1D(const Tensor1D& output, const Tensor1D& target)>& costFunction,
	double learningRate) {
	std::vector<Layer>& layers = neuralNetwork.getLayers();

	assert((layers.size() > 0.0) && "NeuralNetwork is empty!");

	std::vector<PreparePropagateData> preparedData;
	preparedData.reserve(layers.size());

	// Gether prepared propagate data.
	preparedData.emplace_back(layers[0].layer->preparePropagate(input));
	for (size_t i = 1; i < layers.size(); i++)
		preparedData.emplace_back(layers[i].layer->preparePropagate(preparedData[i - 1].output));

	// Calculate the differentiated cost respect to the output
	Tensor1D localCost = costFunction(preparedData[layers.size() - 1].output, target);

	// Push the cost backward in the layers, calculate the gradients and updating the parameters.
	for (int i = (int)layers.size() - 1; i >= 0; i--)
		localCost = layers[i].layer->backPropagate(
			preparedData[i].sum,
			localCost,
			preparedData[i].input, 
			learningRate);

	// The first layer's cost respect to the input.
	return localCost;
}

DRAGON_END
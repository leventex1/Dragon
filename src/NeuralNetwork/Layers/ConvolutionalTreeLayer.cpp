#include "ConvolutionalTreeLayer.h"

DRAGON_BEGIN

ConvolutionalTreeLayer::ConvolutionalTreeLayer() :
	m_InputType({ 0, 0, 0 }), m_OutputType({ 0, 0 }), m_KernelStride(0) { }

ConvolutionalTreeLayer::ConvolutionalTreeLayer(
	size_t inputRows, size_t inputCols, size_t inputDepth,
	size_t kernelRows, size_t kernelCols, size_t kernelCountPerInput,
	size_t kernelStride,
	const std::function<double()>& initFunction,
	const ActivationFunction& activation)
	:
	m_InputType({ inputRows, inputCols, inputDepth }),
	m_OutputType(
		{ calcConvParamsAfter(inputRows, kernelRows, kernelStride),
		calcConvParamsAfter(inputCols, kernelCols, kernelStride),
		inputDepth * kernelCountPerInput 
		}),
	m_KernelStride(kernelStride), 
	BaseLayer(activation) {
	assert((m_InputType.parameters[0] % kernelRows == 0 && m_InputType.parameters[1] % kernelCols == 0) &&
		"Kernel parameters not match with input parameters, loses some information, not supported yet! \
		It make unexcepted behaviour at backpropagation!");	// TODO: support information loss.
	assert((kernelRows == kernelCols) &&
		"Not supported different kernel parameters!");	// TODO: add different kernel and stride parameter support (backpropagate)

	m_Kernels = Tensor3D(kernelCountPerInput * inputDepth, kernelRows, kernelCols, initTensor(
		kernelCountPerInput * inputDepth * kernelRows * kernelCols, initFunction));

	m_Biases = Tensor3D(
		m_OutputType.parameters[2], m_OutputType.parameters[0], m_OutputType.parameters[1],
		initTensor(m_OutputType.getParameterCount(), initFunction));
}

ConvolutionalTreeLayer::ConvolutionalTreeLayer(const ConvolutionalTreeLayer& other) :
	m_InputType(other.m_InputType), m_OutputType(other.m_OutputType),
	m_KernelStride(other.m_KernelStride),
	m_Kernels(other.m_Kernels),
	m_Biases(other.m_Biases),
	BaseLayer(other.m_Activation) { }

ConvolutionalTreeLayer::ConvolutionalTreeLayer(ConvolutionalTreeLayer&& other) noexcept :
	m_InputType(other.m_InputType), m_OutputType(other.m_OutputType),
	m_KernelStride(other.m_KernelStride),
	m_Kernels(std::move(other.m_Kernels)),
	m_Biases(std::move(other.m_Biases)),
	BaseLayer(other.m_Activation) { }

Tensor1D ConvolutionalTreeLayer::feedForward(const Tensor1D& input) const {
	assert((input.getCount() == m_InputType.getParameterCount()) && 
		"Invalid input parameters!");
	// Convert input into formated input.
	Tensor3D working = Tensor3D(
		m_InputType.parameters[2], m_InputType.parameters[0], m_InputType.parameters[1], input);

	// Create a Tensor3D with an allocated memory with the size of the outputtype count,
	// it's just allocate the memory, still hold junk.
	Tensor3D output = Tensor3D(new double[m_OutputType.getParameterCount()], 
		m_OutputType.parameters[2], m_OutputType.parameters[0], m_OutputType.parameters[1]);


	// Calculate the number of output per input.
	size_t kernelCountPerInput = m_OutputType.parameters[2] / m_InputType.parameters[2];

	// Calculate the layer offsets.
	size_t inputLayerCount = m_InputType.parameters[0] * m_InputType.parameters[1];
	size_t outputLayerCount = m_OutputType.parameters[0] * m_OutputType.parameters[1];
	size_t kernelLayerCount = m_Kernels.getRows() * m_Kernels.getCols();

	// Go through the kernel depth and output depth.
	for (size_t i = 0; i < m_OutputType.parameters[2]; i++) {

		size_t inputIndex = size_t(i / kernelCountPerInput);
		
		// Create watcher tensors for inputs, outputs and kernels, initially they are Tensor3D,
		// to calculate the convolution we need Tensor2D.
		Tensor2D inputWatcher = Tensor2D(m_InputType.parameters[0], m_InputType.parameters[1],
			Tensor(working.getData() + inputIndex * inputLayerCount, true));
		Tensor2D outputWatcher = Tensor2D(m_OutputType.parameters[0], m_OutputType.parameters[1],
			Tensor(output.getData() + i * outputLayerCount, true));
		Tensor2D kernelWatcher = Tensor2D(m_Kernels.getRows(), m_Kernels.getCols(),
			Tensor((double*)(m_Kernels.getData() + i * kernelLayerCount), true));

		convolution(outputWatcher, inputWatcher, kernelWatcher, m_KernelStride);
	}

	output.add(m_Biases);
	output.manipul(m_Activation.getActivation());
	return Tensor1D(m_OutputType.getParameterCount(), std::move(output));
}

Tensor1D ConvolutionalTreeLayer::backPropagate(
	Tensor1D& sumsAfter,
	Tensor1D& costAfter,
	Tensor1D& activationsBefore,
	double learningRate) {
	assert((sumsAfter.getCount() == costAfter.getCount() &&
		sumsAfter.getCount() == m_OutputType.getParameterCount()) &&
		"Invalid cost and sums parameters!");
	assert((activationsBefore.getCount() == m_InputType.getParameterCount()) &&
		"Invalid before activation parameters!");

	// Local gradient respect to the output
	sumsAfter.manipul(m_Activation.getActivationDiff()).mult(costAfter);

	// Calculate the memory need for the local gradient respect to the input and the kernel gradient.
	Tensor3D costBefore = Tensor3D(m_InputType.parameters[2], m_InputType.parameters[0], m_InputType.parameters[1], 0.0);

	Tensor3D kernelGradient = Tensor3D(new double[m_Kernels.getDepth() * m_Kernels.getRows() * m_Kernels.getCols()],
		m_Kernels.getDepth(), m_Kernels.getRows(), m_Kernels.getCols());

	// Calculate the number of output per input.
	size_t kernelCountPerInput = m_OutputType.parameters[2] / m_InputType.parameters[2];

	// Calculate the layer offsets.
	size_t inputLayerCount = m_InputType.parameters[0] * m_InputType.parameters[1];
	size_t costLayerCount = m_OutputType.parameters[0] * m_OutputType.parameters[1];
	size_t kernelLayerCount = m_Kernels.getRows() * m_Kernels.getCols();

	// Go through the cost depth.
	for (size_t i = 0; i < m_OutputType.parameters[2]; i++) {

		size_t inputIndex = size_t(i / kernelCountPerInput);

		// Create watcher tensors for kernel, cost, input and localcostbefore, initially they are Tensor3D,
		// to calculate the convolution we need Tensor2D.
		Tensor2D kernelGradientWatcher = Tensor2D(m_Kernels.getRows(), m_Kernels.getCols(),
			Tensor(kernelGradient.getData() + i * kernelLayerCount, true));

		Tensor2D kernelWathcer = Tensor2D(m_Kernels.getRows(), m_Kernels.getCols(),
			Tensor(m_Kernels.getData() + i * kernelLayerCount, true));

		Tensor2D costWatcher = Tensor2D(m_OutputType.parameters[0], m_OutputType.parameters[1],
			Tensor(sumsAfter.getData() + i * costLayerCount, true));

		Tensor2D inputWatcher = Tensor2D(m_InputType.parameters[0], m_InputType.parameters[1],
			Tensor(activationsBefore.getData() + inputIndex * inputLayerCount, true));

		Tensor2D costBeforeWatcher = Tensor2D(m_InputType.parameters[0], m_InputType.parameters[1],
			Tensor(costBefore.getData() + inputIndex * inputLayerCount, true));

		Tensor2D formatedCost;
		if (m_KernelStride > 1)
			formatedCost = scaleByStride(costWatcher, m_KernelStride);

		// Calculate the kernel gradient tensor, It's the convolution between the input and the formated cost.
		convolution(
			kernelGradientWatcher,
			inputWatcher, 
			(m_KernelStride > 1) ? formatedCost : costWatcher,
			1);

		costBeforeWatcher.add(convolution(
				padding((m_KernelStride > 1) ? formatedCost : costWatcher, m_Kernels.getRows() - 1, 0.0),
				reverse(kernelWathcer), 
				1));
	}

	m_Kernels.sub(kernelGradient.mult(learningRate));
	m_Biases.sub(sumsAfter.mult(learningRate));
	return Tensor1D(m_InputType.getParameterCount(), std::move(costBefore));
}

PreparePropagateData ConvolutionalTreeLayer::preparePropagate(const Tensor1D& input) const {
	assert((input.getCount() == m_InputType.getParameterCount()) && 
		"Invalid input parameters!");
	PreparePropagateData pData;

	Tensor3D working = Tensor3D(
		m_InputType.parameters[2], m_InputType.parameters[0], m_InputType.parameters[1], input);

	Tensor3D output = Tensor3D(new double[m_OutputType.getParameterCount()],
		m_OutputType.parameters[2], m_OutputType.parameters[0], m_OutputType.parameters[1]);


	size_t kernelCountPerInput = m_OutputType.parameters[2] / m_InputType.parameters[2];

	size_t inputLayerCount = m_InputType.parameters[0] * m_InputType.parameters[1];
	size_t outputLayerCount = m_OutputType.parameters[0] * m_OutputType.parameters[1];
	size_t kernelLayerCount = m_Kernels.getRows() * m_Kernels.getCols();

	for (size_t i = 0; i < m_OutputType.parameters[2]; i++) {

		size_t inputIndex = size_t(i / kernelCountPerInput);

		Tensor2D inputWatcher = Tensor2D(m_InputType.parameters[0], m_InputType.parameters[1],
			Tensor(working.getData() + inputIndex * inputLayerCount, true));
		Tensor2D outputWatcher = Tensor2D(m_OutputType.parameters[0], m_OutputType.parameters[1],
			Tensor(output.getData() + i * outputLayerCount, true));
		Tensor2D kernelWatcher = Tensor2D(m_Kernels.getRows(), m_Kernels.getCols(),
			Tensor((double*)(m_Kernels.getData() + i * kernelLayerCount), true));

		convolution(outputWatcher, inputWatcher, kernelWatcher, m_KernelStride);
	}

	pData.input = input;
	output.add(m_Biases);
	pData.sum = Tensor1D(m_OutputType.getParameterCount(), output);
	output.manipul(m_Activation.getActivation());
	pData.output = Tensor1D(m_OutputType.getParameterCount(), std::move(output));

	return pData;
}

std::string ConvolutionalTreeLayer::toString() const {
	// Create a string stream. 
	std::stringstream ss;

	ss << m_InputType.parameters[0] << " ";
	ss << m_InputType.parameters[1] << " ";
	ss << m_InputType.parameters[2] << " ";

	ss << m_OutputType.parameters[0] << " ";
	ss << m_OutputType.parameters[0] << " ";
	ss << m_OutputType.parameters[0] << " ";

	ss << m_KernelStride << " ";

	for (size_t i = 0; i < m_Kernels.getCount(); i++)
		ss << m_Kernels.getData()[i] << " ";

	for (size_t i = 0; i < m_Biases.getCount(); i++)
		ss << m_Biases.getData()[i] << " ";

	return ss.str();
}

void ConvolutionalTreeLayer::fromString(const std::string& rawString) {
	std::stringstream ss(rawString);

	ss >> m_InputType.parameters[0];
	ss >> m_InputType.parameters[1];
	ss >> m_InputType.parameters[2];

	ss >> m_OutputType.parameters[0];
	ss >> m_OutputType.parameters[1];
	ss >> m_OutputType.parameters[2];

	ss >> m_KernelStride;

	size_t rows = m_InputType.parameters[0] - (m_OutputType.parameters[0] - 1) * m_KernelStride;
	size_t cols = m_InputType.parameters[1] - (m_OutputType.parameters[1] - 1) * m_KernelStride;

	m_Kernels = Tensor3D(m_OutputType.parameters[2], rows, cols, 0.0);

	m_Biases = Tensor3D(m_OutputType.parameters[2], m_OutputType.parameters[0], m_OutputType.parameters[1], 0.0);

	for (size_t i = 0; i < m_Kernels.getCount(); i++)
		ss >> m_Kernels.getData()[i];

	for (size_t i = 0; i < m_Biases.getCount(); i++)
		ss >> m_Biases.getData()[i];

}

DRAGON_END
#include "ConvolutionalLayer.h"

DRAGON_BEGIN

ConvolutionalLayer::ConvolutionalLayer() :
	m_InputType({ 0, 0, 0 }), m_OutputType({ 0, 0, 0 }), m_KernelStride(0) { }

ConvolutionalLayer::ConvolutionalLayer(
	size_t inputRows, size_t inputCols, size_t inputDepth,
	size_t kernelRows, size_t kernelCols, size_t kernelCount,
	size_t kernelStride,
	const std::function<double()>& initFunction,
	const ActivationFunction& activation)
	:
	m_InputType({ inputRows, inputCols, inputDepth }),
	m_OutputType(
		{ calcConvParamsAfter(inputRows, kernelRows, kernelStride),
		calcConvParamsAfter(inputCols, kernelCols, kernelStride),
		kernelCount
		}),
	m_KernelStride(kernelStride),
	BaseLayer(activation) {
	//assert(((inputRows - kernelStride) % kernelRows == 0 && inputCols % kernelCols == 0) &&
	//	"Kernel parameters not match with input parameters, loses some information, not supported yet! \
	//	It make unexcepted behaviour at backpropagation!");	// TODO: support information loss.
	assert((kernelRows == kernelCols) &&
		"Not supported different kernel parameters!");	// TODO: add different kernel and stride parameter support (backpropagate)

	m_Kernels = Tensor3D(kernelCount * inputDepth, kernelRows, kernelCols, initTensor(
		kernelCount * inputDepth * kernelRows * kernelCols, initFunction));

	m_Biases = Tensor3D(
		m_OutputType.parameters[2], m_OutputType.parameters[0], m_OutputType.parameters[1],
		initTensor(m_OutputType.getParameterCount(), initFunction));
}

ConvolutionalLayer::ConvolutionalLayer(const ConvolutionalLayer& other) :
	m_InputType(other.m_InputType), m_OutputType(other.m_OutputType),
	m_KernelStride(other.m_KernelStride),
	m_Kernels(other.m_Kernels),
	m_Biases(other.m_Biases),
	BaseLayer(other.m_Activation) { }

ConvolutionalLayer::ConvolutionalLayer(ConvolutionalLayer&& other) noexcept :
	m_InputType(other.m_InputType), m_OutputType(other.m_OutputType),
	m_KernelStride(other.m_KernelStride),
	m_Kernels(std::move(other.m_Kernels)),
	m_Biases(std::move(other.m_Biases)),
	BaseLayer(other.m_Activation) { }

Tensor1D ConvolutionalLayer::feedForward(const Tensor1D& input) const {
	assert((input.getCount() == m_InputType.getParameterCount()) &&
		"Invalid input parameters!");
	// Convert input into formated input.
	Tensor3D working = Tensor3D(
		m_InputType.parameters[2], m_InputType.parameters[0], m_InputType.parameters[1], input);

	// Create a output Tensor3D.
	Tensor3D output = Tensor3D(m_OutputType.parameters[2], m_OutputType.parameters[0], m_OutputType.parameters[1], 0.0);

	// Calculate the layer offsets.
	size_t inputLayerCount = m_InputType.parameters[0] * m_InputType.parameters[1];
	size_t outputLayerCount = m_OutputType.parameters[0] * m_OutputType.parameters[1];
	size_t kernelLayerCount = m_Kernels.getRows() * m_Kernels.getCols();

	// Go through the output depth
	for (size_t i = 0; i < m_OutputType.parameters[2]; i++) {

		// Create the output watcher.
		Tensor2D outputWatcher = Tensor2D(m_OutputType.parameters[0], m_OutputType.parameters[1],
			Tensor(output.getData() + i * outputLayerCount, true));

		// Go through the input depth and calculate the sum weighted sums.
		for (size_t k = 0; k < m_InputType.parameters[2]; k++) {

			// Create the input and kernel watchers.
			Tensor2D inputWatcher = Tensor2D(m_InputType.parameters[0], m_InputType.parameters[1],
				Tensor(working.getData() + k * inputLayerCount, true));
			Tensor2D kernelWatcher = Tensor2D(m_Kernels.getRows(), m_Kernels.getCols(),
				Tensor((double*)(m_Kernels.getData() + (i * m_InputType.parameters[2] + k) * kernelLayerCount), true));

			// Calculate the convolution between the input and the kernel watcher and adds to the output tensor.
			outputWatcher.add(std::move(convolution(inputWatcher, kernelWatcher, m_KernelStride)));
		}
	}

	output.add(m_Biases);
	output.manipul(m_Activation.getActivation());
	return Tensor1D(m_OutputType.getParameterCount(), std::move(output));
}

Tensor1D ConvolutionalLayer::backPropagate(
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

	// Calculate the layer offsets.
	size_t inputLayerCount = m_InputType.parameters[0] * m_InputType.parameters[1];
	size_t costLayerCount = m_OutputType.parameters[0] * m_OutputType.parameters[1];
	size_t kernelLayerCount = m_Kernels.getRows() * m_Kernels.getCols();

	// Go through the cost depth.
	for (size_t i = 0; i < m_OutputType.parameters[2]; i++) {
		
		// Create the cost watcher.
		Tensor2D costWatcher = Tensor2D(m_OutputType.parameters[0], m_OutputType.parameters[1],
			Tensor(sumsAfter.getData() + i * costLayerCount, true));
		
		// Create the formated cost if needed.(if we use stide other than 1)
		Tensor2D formatedCost;
		if (m_KernelStride > 1)
			formatedCost = scaleByStride(costWatcher, m_KernelStride);

		// Go throuth the input depth.
		for (size_t k = 0; k < m_InputType.parameters[2]; k++) {

			// Kernel, input and gradients watcher
			Tensor2D kernelGradientWatcher = Tensor2D(m_Kernels.getRows(), m_Kernels.getCols(),
				Tensor(kernelGradient.getData() + (i * m_InputType.parameters[2] + k) * kernelLayerCount, true));

			Tensor2D kernelWathcer = Tensor2D(m_Kernels.getRows(), m_Kernels.getCols(),
				Tensor(m_Kernels.getData() + (i * m_InputType.parameters[2] + k) * kernelLayerCount, true));	
			
			Tensor2D inputWatcher = Tensor2D(m_InputType.parameters[0], m_InputType.parameters[1],
				Tensor(activationsBefore.getData() + k * inputLayerCount, true));
			
			Tensor2D costBeforeWatcher = Tensor2D(m_InputType.parameters[0], m_InputType.parameters[1],
				Tensor(costBefore.getData() + k * inputLayerCount, true));
			

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
	}

	m_Kernels.sub(kernelGradient.mult(learningRate));
	m_Biases.sub(sumsAfter.mult(learningRate));
	return Tensor1D(m_InputType.getParameterCount(), std::move(costBefore));
}

PreparePropagateData ConvolutionalLayer::preparePropagate(const Tensor1D& input) const {
	assert((input.getCount() == m_InputType.getParameterCount()) &&
		"Invalid input parameters!");
	PreparePropagateData pData;

	Tensor3D working = Tensor3D(
		m_InputType.parameters[2], m_InputType.parameters[0], m_InputType.parameters[1], input);

	Tensor3D output = Tensor3D(m_OutputType.parameters[2], m_OutputType.parameters[0], m_OutputType.parameters[1], 0.0);

	size_t inputLayerCount = m_InputType.parameters[0] * m_InputType.parameters[1];
	size_t outputLayerCount = m_OutputType.parameters[0] * m_OutputType.parameters[1];
	size_t kernelLayerCount = m_Kernels.getRows() * m_Kernels.getCols();

	for (size_t i = 0; i < m_OutputType.parameters[2]; i++) {

		Tensor2D outputWatcher = Tensor2D(m_OutputType.parameters[0], m_OutputType.parameters[1],
			Tensor(output.getData() + i * outputLayerCount, true));

		for (size_t k = 0; k < m_InputType.parameters[2]; k++) {

			Tensor2D inputWatcher = Tensor2D(m_InputType.parameters[0], m_InputType.parameters[1],
				Tensor(working.getData() + k * inputLayerCount, true));
			Tensor2D kernelWatcher = Tensor2D(m_Kernels.getRows(), m_Kernels.getCols(),
				Tensor((double*)(m_Kernels.getData() + (i * m_InputType.parameters[2] + k) * kernelLayerCount), true));

			outputWatcher.add(std::move(convolution(inputWatcher, kernelWatcher, m_KernelStride)));
		}
	}

	pData.input = input;
	output.add(m_Biases);
	pData.sum = Tensor1D(m_OutputType.getParameterCount(), output);
	output.manipul(m_Activation.getActivation());
	pData.output = Tensor1D(m_OutputType.getParameterCount(), std::move(output));
	
	return pData;
}

std::string ConvolutionalLayer::toString() const {
	std::stringstream ss;
	ss << std::fixed << std::setprecision(8);
	
	// Put input/output dimensions
	ss << m_InputType.parameters[0] << " ";
	ss << m_InputType.parameters[1] << " ";
	ss << m_InputType.parameters[2] << " ";

	ss << m_OutputType.parameters[0] << " ";
	ss << m_OutputType.parameters[1] << " ";
	ss << m_OutputType.parameters[2] << " ";

	// Put kernel stride.
	ss << m_KernelStride << " ";

	// Put kernel data.
	for (size_t i = 0; i < m_Kernels.getCount(); i++) {
		ss << m_Kernels.getData()[i] << " ";
	}
	// Put bias data.
	for (size_t i = 0; i < m_Biases.getCount(); i++) {
		ss << m_Biases.getData()[i] << " ";
	}

	return ss.str();
}

void ConvolutionalLayer::fromString(const std::string& rawString) {
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

	m_Kernels = Tensor3D(m_OutputType.parameters[2] * m_InputType.parameters[2], rows, cols, 0.0);

	m_Biases = Tensor3D(m_OutputType.parameters[2], m_OutputType.parameters[0], m_OutputType.parameters[1], 0.0);

	for (size_t i = 0; i < m_Kernels.getCount(); i++) {
		ss >> m_Kernels.getData()[i];
	}
	for (size_t i = 0; i < m_Biases.getCount(); i++) {
		ss >> m_Biases.getData()[i];
	}

}

DRAGON_END
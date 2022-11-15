#include "PoolingLayer.h"

DRAGON_BEGIN

PoolingLayer::PoolingLayer() : m_InputType({ 0, 0, 0 }) { }

PoolingLayer::PoolingLayer(
	size_t inputRows, size_t inputCols, size_t inputDepth,
	size_t kernelRows, size_t kernelCols,
	const std::function<double(const std::vector<double>& values)> poolingFunc,
	const std::function<std::vector<double>(const std::vector<double>& values, double value)> poolingFuncDiff) 
	: 
	m_InputType({ inputRows, inputCols, inputDepth }),
	m_KernelRows(kernelRows), m_KernelCols(kernelCols),
	m_PoolingFunction(poolingFunc),
	m_PoolingFunctionDiff(poolingFuncDiff),
	BaseLayer(sigmoid()) {
	assert((m_InputType.parameters[0] % kernelRows == 0 && m_InputType.parameters[1] % kernelCols == 0) &&
		"Kernel parameters not match with input parameters, loses some information, not supported yet! \
		It make unexcepted behaviour at backpropagation!");	// TODO: support information loss.
}

PoolingLayer::PoolingLayer(const PoolingLayer& other) :
	m_InputType(other.m_InputType),
	m_KernelRows(other.m_KernelRows), m_KernelCols(other.m_KernelCols),
	m_PoolingFunction(other.m_PoolingFunction), m_PoolingFunctionDiff(other.m_PoolingFunctionDiff),
	BaseLayer(sigmoid()) { }

PoolingLayer::PoolingLayer(PoolingLayer&& other) noexcept:
	m_InputType(other.m_InputType),
	m_KernelRows(other.m_KernelRows), m_KernelCols(other.m_KernelCols),
	m_PoolingFunction(other.m_PoolingFunction), m_PoolingFunctionDiff(other.m_PoolingFunctionDiff),
	BaseLayer(sigmoid()) { }

Tensor1D PoolingLayer::feedForward(const Tensor1D& input) const {
	assert((input.getCount() == m_InputType.getParameterCount()) && "Invalid input parameters!");

	Tensor3D working = Tensor3D(
		m_InputType.parameters[2], m_InputType.parameters[0], m_InputType.parameters[1], input);

	// Calculate the output parameters.
	size_t row = calcConvParamsAfter(m_InputType.parameters[0], m_KernelRows, m_KernelRows);
	size_t col = calcConvParamsAfter(m_InputType.parameters[1], m_KernelCols, m_KernelCols);

	// Make empty output tensor.
	Tensor3D output = Tensor3D(new double[row * col * m_InputType.parameters[2]],
		m_InputType.parameters[2], row, col);

	// Go throuth the output depth.
	for (size_t k = 0; k < m_InputType.parameters[2]; k++) {

		// Go throuth the output tensor2D.
		for (size_t i = 0; i < row; i++)
			for (size_t j = 0; j < col; j++) {

				// Get the values from the kernel space span.
				std::vector<double> values;
				values.reserve(m_KernelRows * m_KernelCols);
				for (size_t x = 0; x < m_KernelRows; x++)
					for (size_t y = 0; y < m_KernelCols; y++)
						values.emplace_back(working.at(i * m_KernelRows + x, j * m_KernelCols + y, k));

				// Calculate the output value.
				output.at(i, j, k) = m_PoolingFunction(values);
			}
	}

	return Tensor1D(row * col * m_InputType.parameters[2], std::move(output));
}

Tensor1D PoolingLayer::backPropagate(
	Tensor1D& sumsAfter,
	Tensor1D& costAfter,
	Tensor1D& activationsBefore,
	double learningRate) {

	// TODO: make it cleaner.

	size_t row = calcConvParamsAfter(m_InputType.parameters[0], m_KernelRows, m_KernelRows);
	size_t col = calcConvParamsAfter(m_InputType.parameters[1], m_KernelCols, m_KernelCols);

	Tensor3D workingInput = Tensor3D(m_InputType.parameters[2], m_InputType.parameters[0], m_InputType.parameters[1],
		activationsBefore);
	Tensor3D workingCostAfter = Tensor3D(m_InputType.parameters[2], row, col, costAfter);

	Tensor3D costBefore = Tensor3D(m_InputType.parameters[2], m_InputType.parameters[0], m_InputType.parameters[1], 0.0);

	for (size_t k = 0; k < m_InputType.parameters[2]; k++) {

		for (size_t i = 0; i < row; i++)
			for (size_t j = 0; j < col; j++) {

				// Get the values from the kernel space span.
				std::vector<double> values;
				values.reserve(m_KernelRows * m_KernelCols);
				for (size_t x = 0; x < m_KernelRows; x++)
					for (size_t y = 0; y < m_KernelCols; y++)
						values.emplace_back(workingInput.at(i * m_KernelRows + x, j * m_KernelCols + y, k));

				std::vector<double> result = m_PoolingFunctionDiff(values, workingCostAfter.at(i, j, k));

				for (size_t x = 0; x < m_KernelRows; x++)
					for (size_t y = 0; y < m_KernelCols; y++)
						costBefore.at(i * m_KernelRows + x, j * m_KernelCols + y, k) = result.at(x * m_KernelCols + y);

			}
	}
	return Tensor1D(m_InputType.getParameterCount(), std::move(costBefore));
}

PreparePropagateData PoolingLayer::preparePropagate(const Tensor1D& input) const {
	assert((input.getCount() == m_InputType.getParameterCount()) && "Invalid input parameters!");

	PreparePropagateData pData;

	Tensor3D working = Tensor3D(
		m_InputType.parameters[2], m_InputType.parameters[0], m_InputType.parameters[1], input);

	size_t row = calcConvParamsAfter(m_InputType.parameters[0], m_KernelRows, m_KernelRows);
	size_t col = calcConvParamsAfter(m_InputType.parameters[1], m_KernelCols, m_KernelCols);

	Tensor3D output = Tensor3D(new double[row * col * m_InputType.parameters[2]],
		m_InputType.parameters[2], row, col);

	for (size_t k = 0; k < m_InputType.parameters[2]; k++) {

		Tensor2D outputWatcher(row, col, Tensor(output.getData() + k * row * col, true));

		for (size_t i = 0; i < row; i++)
			for (size_t j = 0; j < col; j++) {

				std::vector<double> values;
				values.reserve(m_KernelRows * m_KernelCols);
				for (size_t x = 0; x < m_KernelRows; x++)
					for (size_t y = 0; y < m_KernelCols; y++)
						values.emplace_back(working.at(i * m_KernelRows + x, j * m_KernelCols + y, k));

				outputWatcher.at(i, j) = m_PoolingFunction(values);
			}
	}
	pData.input = input;
	pData.output = std::move(Tensor1D(m_InputType.parameters[2] * row * col, std::move(output)));
	return pData;
}

std::string PoolingLayer::toString() const {
	std::stringstream ss;
	ss << std::fixed << std::setprecision(8);

	// Put parameters.
	ss << m_InputType.parameters[0] << " ";
	ss << m_InputType.parameters[1] << " ";
	ss << m_InputType.parameters[2] << " ";

	// Put kernel dimension.
	ss << m_KernelRows << " ";
	ss << m_KernelCols << " ";

	return ss.str();
}

void PoolingLayer::fromString(const std::string& rawString) {
	std::stringstream ss(rawString);

	// Get parameters.
	ss >> m_InputType.parameters[0];
	ss >> m_InputType.parameters[1];
	ss >> m_InputType.parameters[2];

	// Get kernel dimension.
	ss >> m_KernelRows;
	ss >> m_KernelCols;

	m_PoolingFunction = maxPool;
	m_PoolingFunctionDiff = maxPoolDiff;

}


DRAGON_END
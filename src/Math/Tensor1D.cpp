#include "Tensor1D.h"

DRAGON_BEGIN

Tensor1D::Tensor1D() { }

Tensor1D::Tensor1D(const Tensor1D& other) :
	m_Cols(other.m_Cols), Tensor(other) { }

Tensor1D::Tensor1D(Tensor1D&& other) noexcept :
	m_Cols(other.m_Cols), Tensor(std::move(other)) { }

Tensor1D* Tensor1D::operator=(const Tensor1D& other) {
	m_Cols = other.m_Cols;
	_copy(other.m_Data, getCount());
	return this;
}

Tensor1D* Tensor1D::operator=(Tensor1D&& other) noexcept {
	std::swap(m_Cols, other.m_Cols);
	_swap(std::move(other.m_Data));
	return this;
}

Tensor1D::~Tensor1D() { }

Tensor1D::Tensor1D(size_t cols, precision value) :
	m_Cols(cols), Tensor(cols, value) { }

Tensor1D::Tensor1D(size_t cols, const precision* copyPointer) :
	m_Cols(cols), Tensor(copyPointer, cols) { }

Tensor1D::Tensor1D(precision* assignPointer, size_t cols) :
	m_Cols(cols), Tensor(assignPointer) { }

Tensor1D::Tensor1D(size_t cols, Tensor&& dataTensor) :
	m_Cols(cols), Tensor(std::move(dataTensor)) { }

Tensor1D::Tensor1D(size_t cols, const Tensor& dataTensor) :
	m_Cols(cols), Tensor(dataTensor) {
	assert((getCount() == dataTensor.getCount()) && "Parameter count not match!");
}

Tensor1D::Tensor1D(const std::initializer_list<precision>& initList) :
	m_Cols(initList.size()) {
	m_Data = new precision[m_Cols];
	auto it = initList.begin();
	for (size_t i = 0; i < m_Cols; i++) {
		m_Data[i] = *it;
		it++;
	}
}

void Tensor1D::_deleteParams() { m_Cols = 0; }

DRAGON_END
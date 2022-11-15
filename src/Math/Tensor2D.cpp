#include "Tensor2D.h"

DRAGON_BEGIN

Tensor2D::Tensor2D() { }

Tensor2D::Tensor2D(const Tensor2D& other) :
	m_Rows(other.m_Rows), m_Cols(other.m_Cols), Tensor(other) { }

Tensor2D::Tensor2D(Tensor2D&& other) noexcept :
	m_Rows(other.m_Rows), m_Cols(other.m_Cols), Tensor(std::move(other)) { }

Tensor2D* Tensor2D::operator=(const Tensor2D& other) {
	m_Rows = other.m_Rows;
	m_Cols = other.m_Cols;
	_copy(other.m_Data, getCount());
	return this;
}

Tensor2D* Tensor2D::operator=(Tensor2D&& other) noexcept {
	std::swap(m_Rows, other.m_Rows);
	std::swap(m_Cols, other.m_Cols);
	_swap(std::move(other.m_Data));
	return this;
}

Tensor2D::~Tensor2D() { }

Tensor2D::Tensor2D(size_t rows, size_t cols, precision value) :
	m_Rows(rows), m_Cols(cols), Tensor(rows * cols, value) { }

Tensor2D::Tensor2D(size_t rows, size_t cols, const precision* copyPointer) :
	m_Rows(rows), m_Cols(cols), Tensor(copyPointer, rows * cols) { }

Tensor2D::Tensor2D(precision* assignPointer, size_t rows, size_t cols) : 
	m_Rows(rows), m_Cols(cols), Tensor(assignPointer) { }

Tensor2D::Tensor2D(size_t rows, size_t cols, Tensor&& dataTensor) :
	m_Rows(rows), m_Cols(cols), Tensor(std::move(dataTensor)) { }

Tensor2D::Tensor2D(size_t rows, size_t cols, const Tensor& dataTensor) :
	m_Rows(rows), m_Cols(cols), Tensor(dataTensor) {
	assert((getCount() == dataTensor.getCount()) && "Parameter count not match!");
}

void Tensor2D::_deleteParams() { m_Rows = 0, m_Cols = 0; }

DRAGON_END
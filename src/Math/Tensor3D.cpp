#include "Tensor3D.h"

DRAGON_BEGIN

Tensor3D::Tensor3D() { }

Tensor3D::Tensor3D(const Tensor3D& other) :
	m_Depth(other.m_Depth), m_Rows(other.m_Rows), m_Cols(other.m_Cols), Tensor(other) { }

Tensor3D::Tensor3D(Tensor3D&& other) noexcept :
	m_Depth(other.m_Depth), m_Rows(other.m_Rows), m_Cols(other.m_Cols), Tensor(std::move(other)) { }

Tensor3D* Tensor3D::operator=(const Tensor3D& other) {
	m_Depth = other.m_Depth;
	m_Rows = other.m_Rows;
	m_Cols = other.m_Cols;
	_copy(other.m_Data, getCount());
	return this;
}

Tensor3D* Tensor3D::operator=(Tensor3D&& other) noexcept {
	std::swap(m_Depth, other.m_Depth);
	std::swap(m_Rows, other.m_Rows);
	std::swap(m_Cols, other.m_Cols);
	_swap(std::move(other.m_Data));
	return this;
}

Tensor3D::~Tensor3D() { }

Tensor3D::Tensor3D(size_t depth, size_t rows, size_t cols, precision value) :
	m_Depth(depth), m_Rows(rows), m_Cols(cols), Tensor(depth * rows* cols, value) { }

Tensor3D::Tensor3D(size_t depth, size_t rows, size_t cols, const precision* copyPointer) :
	m_Depth(depth), m_Rows(rows), m_Cols(cols), Tensor(copyPointer, rows* cols) { }

Tensor3D::Tensor3D(precision* assignPointer, size_t depth, size_t rows, size_t cols) :
	m_Depth(depth), m_Rows(rows), m_Cols(cols), Tensor(assignPointer) { }

Tensor3D::Tensor3D(size_t depth, size_t rows, size_t cols, Tensor&& dataTensor) :
	m_Depth(depth), m_Rows(rows), m_Cols(cols), Tensor(std::move(dataTensor)) { }

Tensor3D::Tensor3D(size_t depth, size_t rows, size_t cols, const Tensor& dataTensor) :
	m_Depth(depth), m_Rows(rows), m_Cols(cols), Tensor(dataTensor) {
	assert((getCount() == dataTensor.getCount()) && "Parameter count not match!");
}

void Tensor3D::_deleteParams() { m_Depth = 0; m_Rows = 0; m_Cols = 0; }

DRAGON_END
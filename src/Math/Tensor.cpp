#include "Tensor.h"

DRAGON_BEGIN

Tensor::Tensor(precision* assignPointer, bool watcher /* = false*/) :
	m_Data(assignPointer), m_Watcher(watcher) { }

Tensor::Tensor(const precision* copyPointer, size_t count) {
	_copy(copyPointer, count);
}

Tensor::Tensor(const Tensor& other) {
	_copy(other.m_Data, other.getCount());
}

Tensor::Tensor(Tensor&& other) noexcept :
	m_Data(other.m_Data), m_Watcher(other.m_Watcher) {
	other.m_Data = nullptr;
	other.m_Watcher = false;
	//other._deleteParams();
}

Tensor::~Tensor() {
	_clear();
}

Tensor::Tensor(size_t count, precision value) {
	_allocate(count);
	for (size_t i = 0; i < count; i++)
		m_Data[i] = value;
	//memcpy(m_Data, &value, count * sizeof(precision));
}

void Tensor::_clear() {
	if (m_Data && !m_Watcher)
		delete[] m_Data;
}

void Tensor::_copy(const precision* other, size_t count) {
	_allocate(count);
	for (size_t i = 0; i < count; i++)
		m_Data[i] = other[i];
}

void Tensor::_allocate() {
	assert((m_Watcher == false) && "You can't allocate memory in a watcher tensor!");
	_clear();
	m_Data = new precision[getCount()];
}

void Tensor::_allocate(size_t count) {
	assert((m_Watcher == false) && "You can't allocate memory in a watcher tensor!");
	_clear();
	m_Data = new precision[count];
}

void Tensor::_swap(precision*&& data) {
	assert((m_Watcher == false) && "You can't swap in a watcher tensor!");
	precision* temp = m_Data;
	m_Data = data;
	data = temp;
}

void Tensor::_deleteParams() { }

Tensor& Tensor::add(const Tensor& other) {
	assert((getCount() == other.getCount()) && "Parameter count not match!");
		for (size_t i = 0; i < getCount(); i++)
			m_Data[i] += other.m_Data[i];
	return *this;
}

Tensor& Tensor::sub(const Tensor& other) {
	assert((getCount() == other.getCount()) && "Parameter count not match!");
	for (size_t i = 0; i < getCount(); i++)
			m_Data[i] -= other.m_Data[i];
	return *this;
}

Tensor& Tensor::mult(const Tensor& other) {
	assert((getCount() == other.getCount()) && "Parameter count not match!");
		for (size_t i = 0; i < getCount(); i++)
			m_Data[i] *= other.m_Data[i];
	return *this;
}

Tensor& Tensor::div(const Tensor& other) {
	assert((getCount() == other.getCount()) && "Parameter count not match!");
	for (size_t i = 0; i < getCount(); i++)
			m_Data[i] /= other.m_Data[i];
	return *this;
}

Tensor& Tensor::add(precision value) {
	for (size_t i = 0; i < getCount(); i++)
		m_Data[i] += value;
	return *this;
}

Tensor& Tensor::sub(precision value) {
	for (size_t i = 0; i < getCount(); i++)
		m_Data[i] -= value;
	return *this;
}

Tensor& Tensor::mult(precision value) {
	for (size_t i = 0; i < getCount(); i++)
		m_Data[i] *= value;
	return *this;
}

Tensor& Tensor::div(precision value) {
	for (size_t i = 0; i < getCount(); i++)
		m_Data[i] /= value;
	return *this;
}

Tensor& Tensor::manipul(const std::function<void(precision&)>& function) {
	for (size_t i = 0; i < getCount(); i++)
		function(m_Data[i]);

	return *this;
}

DRAGON_END
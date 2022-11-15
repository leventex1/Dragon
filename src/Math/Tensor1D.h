#pragma once

#include "Tensor.h"

DRAGON_BEGIN

/// <summary>
/// Derived class from Tensor.
/// This is basicly a mathematical Vector.
/// Member variables is columns.
/// </summary>
class DRAGON_API Tensor1D : public Tensor {
protected:
	size_t m_Cols = 0;

public:
	Tensor1D();
	Tensor1D(const Tensor1D& other);
	Tensor1D(Tensor1D&& other) noexcept;
	Tensor1D* operator=(const Tensor1D& other);
	Tensor1D* operator=(Tensor1D&& other) noexcept;
	~Tensor1D();

	Tensor1D(const std::initializer_list<precision>& initList);
	Tensor1D(size_t cols, precision value);
	Tensor1D(size_t cols, const precision* copyPointer);
	Tensor1D(precision* assignPointer, size_t cols);
	Tensor1D(size_t cols, Tensor&& dataTensor);
	Tensor1D(size_t cols, const Tensor& dataTensor);


	inline size_t getCount() const override { return m_Cols; }

private:
	void _deleteParams() override;
public:
	inline size_t getCols() const { return m_Cols; }

	inline const precision& at(size_t i) const { return m_Data[i]; }
	inline precision& at(size_t i) { return m_Data[i]; }
};

DRAGON_END
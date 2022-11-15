#pragma once

#include "Tensor.h"

DRAGON_BEGIN

/// <summary>
/// Derived class from Tensor.
/// This is basicly a mathematical matrix. 
/// This class is good for 2D signal and image processing (in a toy sense).
/// Member variables are the rows and columns.
/// </summary>
class DRAGON_API Tensor2D : public Tensor {
protected:
	size_t m_Rows = 0;
	size_t m_Cols = 0;

public:
	Tensor2D();
	Tensor2D(const Tensor2D& other);
	Tensor2D(Tensor2D&& other) noexcept;
	Tensor2D* operator=(const Tensor2D& other);
	Tensor2D* operator=(Tensor2D&& other) noexcept;
	~Tensor2D();

	Tensor2D(size_t rows, size_t cols, precision value);
	Tensor2D(size_t rows, size_t cols, const precision* copyPointer);
	Tensor2D(precision* assignPointer, size_t rows, size_t cols);
	Tensor2D(size_t rows, size_t cols, Tensor&& dataTensor);
	Tensor2D(size_t rows, size_t cols, const Tensor& dataTensor);

	inline size_t getCount() const override { return m_Rows * m_Cols; }

private:
	void _deleteParams() override;
public:

	inline size_t getRows() const { return m_Rows; }
	inline size_t getCols() const { return m_Cols; }

	inline const precision& at(size_t i, size_t j) const { return m_Data[i * m_Cols + j]; }
	inline precision& at(size_t i, size_t j) { return m_Data[i * m_Cols + j]; }
};

DRAGON_END
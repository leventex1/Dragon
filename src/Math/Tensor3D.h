#pragma once

#include "Tensor2D.h"

DRAGON_BEGIN

/// <summary>
/// Derived class from Tensor.
/// Create a 3D Grid like container, Voxel.
/// This class is good for 3D signal and voxel processing (in a toy sense).
/// Member variables are the rows, columns and depths.
/// </summary>
class DRAGON_API Tensor3D : public Tensor {
protected:
	size_t m_Depth = 0;
	size_t m_Rows = 0;
	size_t m_Cols = 0;

public:
	Tensor3D();
	Tensor3D(const Tensor3D& other);
	Tensor3D(Tensor3D&& other) noexcept;
	Tensor3D* operator=(const Tensor3D& other);
	Tensor3D* operator=(Tensor3D&& other) noexcept;
	~Tensor3D();

	Tensor3D(size_t depth, size_t rows, size_t cols, precision value);
	Tensor3D(size_t depth, size_t rows, size_t cols, const precision* copyPointer);
	Tensor3D(precision* assignPointer, size_t depth, size_t rows, size_t cols);
	Tensor3D(size_t depth, size_t rows, size_t cols, Tensor&& dataTensor);
	Tensor3D(size_t depth, size_t rows, size_t cols, const Tensor& dataTensor);

	inline size_t getCount() const override { return m_Rows * m_Cols * m_Depth; }

private:
	void _deleteParams() override;
public:

	inline size_t getRows() const { return m_Rows; }
	inline size_t getCols() const { return m_Cols; }
	inline size_t getDepth() const { return m_Depth; }

	inline const precision& at(size_t i, size_t j, size_t k) const { return m_Data[k * m_Rows * m_Cols + i * m_Rows + j]; }
	inline precision& at(size_t i, size_t j, size_t k) { return m_Data[k * m_Rows * m_Cols + i * m_Rows + j]; }
};

DRAGON_END
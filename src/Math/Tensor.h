#pragma once
#include <iostream>
#include <assert.h>
#include <functional>

#include "../Core.h"

/*
Base class for the tensor objects
*/

#ifndef precision
#define precision double
#endif // !precision

DRAGON_BEGIN

/// <summary>
/// Tensor class is the base class for all types of tensor,
/// types like 2D or 3D or higher, these types only needs to care about the data layout.
/// Has a member variable a pointer to precision data array, 
/// and a bool if the object is just a watcher.
/// Basic methods are manipulations. (Add, subbtract, multiply, divide, lambda funciton)
/// these are element wise operations.
///	If the m_Watcher member(bool) is false the tensor works the same, but if its ture
/// the Tensor destuctor wont delete the pointer.
/// </summary>
class DRAGON_API Tensor {
public:
	Tensor() = default;
	virtual ~Tensor();
	Tensor(const Tensor& other);
	Tensor(Tensor&& other) noexcept;

	Tensor(size_t count, precision value);
	Tensor(const precision* copyPointer, size_t count);

public:
	// For watcher classes
	// Get the pointer from the watcher tensor getData function and use bool wathcer as true
	// than cast this Tensor to the tensor type where you get the data
	// [example] Tensor2D watcher = Tensor2D(3, 3, Tensor(aTensor3DType.getData() + offset, true));
	Tensor(precision* assignPointer, bool watcher = false);
	
public:
	// Rerturns the number of elements that the tensor store.
	virtual size_t getCount() const { assert((false) && "Never use this function in this type!"); return 0; };

public:

	Tensor& add(const Tensor& other);
	Tensor& sub(const Tensor& other);
	Tensor& mult(const Tensor& other);
	Tensor& div(const Tensor& other);
	Tensor& add(precision value);
	Tensor& sub(precision value);
	Tensor& mult(precision value);
	Tensor& div(precision value);

	// Apply the function to every element of the tensor.
	// The function is void and takes a precision reference as parameter(the tensor element).
	Tensor& manipul(const std::function<void(precision&)>& function);

public:

	inline precision*& getData() { return m_Data; }
	inline const precision* getData() const { return m_Data; }
	inline bool isWatcher() const { return m_Watcher; }

protected:
	void _clear();
	void _copy(const precision* other, size_t count);
	void _allocate();
	void _allocate(size_t count);
	void _swap(precision*&& data);
	virtual void _deleteParams();
protected:
	precision* m_Data = nullptr;
	bool m_Watcher = false;

#ifdef USE_CUDA
	bool m_OnDevice = false;
#endif

};

DRAGON_END
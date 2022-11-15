#pragma once
#include <assert.h>

#define DRAGON_VERSION "1.1"

#ifdef DRAGON_BUILD_DLL
#define DRAGON_API __declspec(dllexport)
#else
#define DRAGON_API __declspec(dllimport)
#endif // DRG_BUILD_DLL

#define DRAGON_BEGIN namespace drg {
#define DRAGON_END }

#ifdef __CUDA__ 
	#define USE_CUDA
#endif
#pragma once

#include <random>

#include "../../Core.h"

/*
	Containes some usefull Neural Network layer initializer functions.
*/

DRAGON_BEGIN

template<size_t nInputNodes, size_t nOutputNodes>
double initXavier() {
	std::random_device rg;
	double min = -1.0 / sqrt(nInputNodes), max = 1.0 / sqrt(nInputNodes);
	return min + ((double)rg() / (double)rg.max()) * (max - min);
}

template<size_t nInputNodes, size_t nOutputNodes>
double initNormXavier() {
	std::random_device rg;
	double min = -sqrt(6.0 / double(nInputNodes + nOutputNodes)), max = sqrt(6.0 / double(nInputNodes + nOutputNodes));
	return min + ((double)rg() / (double)rg.max()) * (max - min);
}

template<size_t nInputNodes, size_t nOutputNodes>
double initHe() {
	std::random_device rg;
	double min = -sqrt(2.0 / nInputNodes), max = sqrt(2.0 / nInputNodes);
	return min + ((double)rg() / (double)rg.max()) * (max - min);
}

DRAGON_END
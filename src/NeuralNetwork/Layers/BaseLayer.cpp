#pragma once
#include "BaseLayer.h"

DRAGON_BEGIN

BaseLayer::BaseLayer(
	const ActivationFunction& activation) :
	m_Activation(activation) { }

DRAGON_END
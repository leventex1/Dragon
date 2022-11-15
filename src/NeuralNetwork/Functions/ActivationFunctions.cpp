#include "ActivationFunctions.h"

DRAGON_BEGIN

ActivationFunction::ActivationFunction(
	const std::function<void(double& x)>& activation,
	const std::function<void(double& x)>& activationDiff,
	const std::string& name) :
	m_Activation(activation),
	m_ActivationDiff(activationDiff),
	m_Name(name) {
	assert((!m_Name.empty()) &&
	"Activation function name cannot be empty!");
	assert((m_Name.find('@') == std::string::npos) &&
	"Activation function name cannot contain '@' sign.");
}

namespace activation {
	void sigmoid(double& x) { x = 1.0 / (1.0 + exp(-x)); }
	void sigmoidDiff(double& x) { double tmp = x; sigmoid(tmp); x = tmp * (1.0 - tmp); }
}

ActivationFunction sigmoid()	{ return ActivationFunction(activation::sigmoid, activation::sigmoidDiff, "sigmoid"); }
ActivationFunction relU()		{ return ActivationFunction(activation::relU<0>, activation::relUDiff<0>, "relU"); }
ActivationFunction relU10()		{ return ActivationFunction(activation::relU<10>, activation::relUDiff<10>, "relU10"); }
ActivationFunction relU100()	{ return ActivationFunction(activation::relU<100>, activation::relUDiff<100>, "relU100"); }
ActivationFunction relU500()	{ return ActivationFunction(activation::relU<500>, activation::relUDiff<500>, "relU500"); }

DRAGON_END
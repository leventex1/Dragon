#pragma once
#include <math.h>
#include <functional>
#include <string>

#include "../../Core.h"

DRAGON_BEGIN

// Define a structure of an activation function and it's derivative.
// The getName funciton needed for the model to load and save the model.
class DRAGON_API ActivationFunction {
public:
	ActivationFunction() = default;
	ActivationFunction(
		const std::function<void(double& x)>& activation,
		const std::function<void(double& x)>& activationDiff,
		const std::string& name);
public:

	inline const std::function<void(double& x)>& getActivation() const { return m_Activation; }
	inline const std::function<void(double& x)>& getActivationDiff() const { return m_ActivationDiff; }
	inline const std::string& getName() const { return m_Name; }

private:
	std::function<void(double& x)> m_Activation;
	std::function<void(double& x)> m_ActivationDiff;
	std::string m_Name;
};


// Built in activation functions.
DRAGON_API ActivationFunction sigmoid();
// Returns the normal relU function.
DRAGON_API ActivationFunction relU();
// Returns the relU function with an offset tangent of 1/10.
DRAGON_API ActivationFunction relU10();
// Returns the relU function with an offset tangent of 1/100.
DRAGON_API ActivationFunction relU100();
// Returns the relU function with an offset tangent of 1/500.
DRAGON_API ActivationFunction relU500();

namespace activation {
	// Common Neurla Net activation function.
	// Clips the values betwen -1 and 1. If x <<< -1 -> 0 , x >>> 1 -> 1.
	// Values that are close to 0 are become closly linear.
	DRAGON_API void sigmoid(double& x);
	// Common Neurla Net activation function, the derivative of sigmoid.
	// The values are closly 0 where x <<< -1 or x >>> 1, and if x == 0 it's 1.
	DRAGON_API void sigmoidDiff(double& x);

	// Commont Neural Net activation function.
	// Template parameter ratio is the reciprocal of the function tangent "m"
	// The values x < 0 are multiplied with m, and where x >= 0 are multiplied with 1 + m.
	template<int ratio>
	void relU(double& x) { double m = (ratio) ? 1.0 / (double)ratio : 0.0; x = (x >= 0.0) ? (1.0 + m) * x : m * x; }
	// Commont Neural Net activation function, the derivative of relU.
	// Template parameter ratio is the reciprocal of the function tangent "m"
	// The values x < 0 are become m, else they become 1 + m.
	template<int ratio>
	void relUDiff(double& x) { double m = (ratio) ? 1.0 / (double)ratio : 0.0; x = (x >= 0.0) ? 1.0 + m : m; }
}

DRAGON_END
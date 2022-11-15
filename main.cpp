#include <iostream>
#include <chrono>

#include <Dragon.h>


class Timer {
private:
	std::chrono::time_point<std::chrono::system_clock> m_StartTime;

public:
	Timer() {
		m_StartTime = std::chrono::system_clock::now();
	}

	~Timer() {
		std::chrono::time_point<std::chrono::system_clock>
			endTime = std::chrono::system_clock::now();

		double duration = std::chrono::duration<double>(endTime - m_StartTime).count();

		std::cout << "duration: " << duration << " [ms]" << std::endl;
	}

};

// Model cost function.
drg::Tensor1D costFunction(const drg::Tensor1D& output, const drg::Tensor1D& target) {
	// It should return the cost with respect of the output.
	drg::Tensor1D working = drg::sub(output, target);
	working.mult(2.0);
	return working;
}


int main() {
	std::cout << "Dragon net simple test project." << std::endl;
	drg::DragonTest();


	// Create the layers.
	drg::DenseLayer layer1 = drg::DenseLayer(2, 3, drg::initHe<2, 3>, drg::sigmoid());
	drg::DenseLayer layer2 = drg::DenseLayer(3, 3, drg::initHe<3, 3>, drg::sigmoid());
	drg::DenseLayer layer3 = drg::DenseLayer(3, 1, drg::initHe<3, 1>, drg::sigmoid());

	// Create the model.
	// Add the layer pointers to the model.
	double learning_rate = 1.0;
	drg::Model model;
	model.addLayer(&layer1);
	model.addLayer(&layer2);
	model.addLayer(&layer3);

	// Simple XOR inputs
	/*
		[0, 0] = [0]
		[0, 1] = [1]
		[1, 0] = [1]
		[1, 1] = [0]
	*/
	drg::Tensor2D inputs = drg::Tensor2D(4, 2, 0.0);
	inputs.at(1, 1) = 1.0;
	inputs.at(2, 0) = 1.0;
	inputs.at(3, 0) = 1.0;
	inputs.at(3, 1) = 1.0;

	// XOR targets
	drg::Tensor2D targets = drg::Tensor2D(4, 1, 0.0);
	targets.at(1, 0) = 1.0;
	targets.at(2, 0) = 1.0;


	//---------------------------
	// Test model before training. It should be look like this:
	// ~0.5
	// ~0.5
	// ~0.5
	// ~0.5
	//---------------------------
	std::cout << "Model feedforward before training." << std::endl;
	for (size_t i = 0; i < 4; i++) {
		// Watcher class. Dont copy data just watching it, wont delete data when exit from scope.
		drg::Tensor1D input = drg::Tensor1D(2, drg::Tensor(inputs.getData() + 2 * i, true));
		// Feed the data forward in the network and get output.
		drg::Tensor1D output = drg::feedForward(model, input);
		drg::print(output);
	}


	//---------------------------
	// Start training
	//---------------------------
	for (size_t i = 0; i < 5000; i++) {
		// Go through all 4 training data.
		for (size_t t = 0; t < 4; t++) {
			// Get the input and target data.
			drg::Tensor1D input = drg::Tensor1D(2, drg::Tensor(inputs.getData() + 2 * t, true));
			drg::Tensor1D target = drg::Tensor1D(1, drg::Tensor(targets.getData() + 1 * t, true));

			// Train the model with the backprog algorithm.
			drg::trainModel(model, input, target, costFunction, learning_rate);
		}
	}


	//---------------------------
	// Test model before training. It should be look like this:
	// ~0.0
	// ~1.0
	// ~1.0
	// ~0.0
	//---------------------------
	{
		Timer timer;
		std::cout << "Model feedforward after training." << std::endl;
		for (size_t i = 0; i < 4; i++) {
			// Watcher class. Dont copy data just watching it, wont delete data when exit from scope.
			drg::Tensor1D input = drg::Tensor1D(2, drg::Tensor(inputs.getData() + 2 * i, true));
			// Feed the data forward in the network and get output.
			drg::Tensor1D output = drg::feedForward(model, input);
			drg::print(output);
		}
		std::cout << "4 feed forward time on CPU: ";
	}


	return 0;
} 
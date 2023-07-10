#pragma once

#include "model.h"

#include <cstdint>
#include <vector>

//FFNode = Feed Forward Node
// Constructed as a Fully-connected, feed-forward node/layer


class FFNode : public Node {
public:
	FFNode(
		Model& model,
		std::string name,
		Activation activation,
		uint16_t output_sixe,
		uint16_t input_size,
		);

	// Initialize the parameters of the layer
	// F.T.R. by Jeremy Ong (Direct copy comments: https://github.com/VictorieeMan/fork_cpp_nn_in_a_weekend/blob/152e8cbd361161d8b526c021ef9818ea9dbfe034/src/FFNode.hpp#L21C5-L26C50)
	// Experiment with alternative weight and bias initialization schemes:
	// 1. Try different distributions for the weight
	// 2. Try initializing all weights to zero (why is this suboptimal?)
	// 3. Try initializing all the biases to zero

	void init(rne_t& rne) override;

	//The input vector should have size input_size_
	void forward(num_t* inputs) override;
	//The output vector should have size output_size;
	void reverse(num_t* gradients) override;

	size_t param_count() const noexcept override {
		//Weight matrix entries + biasd entries
		return (input_size_ + 1) * output_size_;
	}

private:
	Activation activation_;
	uint16_t output_size_;
	uint16_t input_size_;

};
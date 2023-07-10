#include "FFNode.h"

#include<algorithm>
#include<cmath>
#include <cstdio>
#include <random>

FFNode::FFNode(
	Model& model,
	std::string name,
	Activation activation,
	uint16_t output_size,
	uint16_t input_size
) 
	: Node{ model, std::move(name) }
	, activation_{activation}
	, output_size_{output_size}
	, input_size_{input_size}
{
	std::printf("%s: %d -> %d\n", name_.c_str(), input_size_, output_size_);

	//The weight parameters of a FF-layer form an n x m - matrix.
	weights_.resize(output_size_ * input_size_);

	//Each node in this later is assigned a bias.
	//(so that zero isn't necessarily mapped to zero.)
	biases_.resize(output_size_);

	//The output of each neuron within the layer is an "activation",
	//in neuroscience lingo.
	activations_.resize(output_size_);

	activation_gradients_.resize(output_size_);
	weight_gradients_.resize(output_size * input_size_);
	bias_gradients_.resize(output_size_);
	input_gradients_.resize(input_size_);
}

void FFNode::init(rne_t& rne) {
	num_t sigma;
	switch (activation_) {
	case Activation::ReLU:
		//Kaiming He, et. al. weight initialization for ReLU networks
		// https://arxiv.org/pdf/1502.01852.pdf
		//
		// Suggests using a normal distribution with variance := 2 / n_in
		sigma = std::sqrt(2.0 / static_cast<num_t>(input_size_));
		break;
	case Activation::Softmax:
	default:
		sigma = std::sqrt(1.0 / static_cast<num_t>(input_size_));
	}

	// NOTE: The C++ standard doesn't guarantee that std::normal_distribution,
	// is consistent across compilers & devices. For production ML code,
	// you should use a library that provides a consistent implementation.
	// Or make a custom implementation, that is deterministic.
	auto dist = std::normal_distribution<num_t>{ 0.0, sigma };
}
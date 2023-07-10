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

}
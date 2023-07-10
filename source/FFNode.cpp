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

	for (num_t& w : weights_) {
		w = dist(rne);
	}
	/*Remember from the model.h file, that rne_t alias is conected to a random
	nnumber generator algorithm. Hence rne are random numbers.*/

	// NOTE: Setting biases to zero is a common practice, as is initializing the
	// bias to a small value (e.g. on the order of 0.01). It is unclear if the
	// latter produces a consistent result over the former, but the thinking is
	// that a non-zero bias will ensure that the neuron always "fires" at the
	// beginning to produce a signal.
	//
	// Here, we initialize all biases to a small number, but the reader should
	// consider experimenting with other approaches.
	for (num_t& b : biases_) {
		b = 0.01;
	}
}

void FFNode::forward(num_t* inputs) {
	//Remember the last input data for backpropogation later.
	last_input_ = inputs;

	for (size_t i = 0; i != output_size_; ++i) {
		//For each output vector, sompute the dot product of the input vector,
		//use weight vector to add the bias.

		num_t z{0.0};
		size_t offset = i * input_size_;

		for (size_t j = 0; j != input_size_; ++j) {
			z += weights_[offset + j] * inputs[j];
		}
		//Add neuron bias
		z += biases_[i];

		switch (activation_) {
		case Activation::ReLU:
			activations_[i] = std::max(z, num_t{ 0.0 });
			break;
		case Activation::Softmax:
		default:
			activations_[i] = std::exp(z);
			break;
		}
	}

	if (activation_ == Activation::Softmax) {
		//Softmax(z)_i = exp(z_i) / sum_j(exp(z_j))
		num_t sum_exp_z{ 0.0 };
		for (size_t i = 0; i != output_size_; ++i) {
			// NOTE: with exploding gradients, it is quite easy for this
			// exponential function to overflow, which will result in NaNs
			// infecting the network.
			sum_exp_z += activations_[i];
		}
	}

	//Forward activation data to all subsequent nodes in the computational graph.
	for(Node* subsequent : subsequents_){
		subsequent->forward(activations_.data());
	}
}
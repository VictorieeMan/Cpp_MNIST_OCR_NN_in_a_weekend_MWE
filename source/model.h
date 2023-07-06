#pragma once

#include <random>

// Declaring type aliases
using num_t = float;		//Default precsision, single 32-bit
using rne_t = std::mt19937; //Random number engine, for double 64-bit use std::mt19937_64


enum class Activation {
	ReLu,
	Softmax
};
/*Enumerations are often used to represent a fixed set of related values in a 
more readable and type-safe way than using integer constants. In this case, 
the Activation enumeration is used to represent the type of
activation function used in the neural network.*/

class Model;

// Base class of computational nodes
class Node {
public:
	Node(Model& model, std::string name);
	virtual ~Node() {};
	/*The virtual keyword indicates that the destructor is virtual, which means
	that it can be overridden by derived classes. This allows derived classes to
	provide their own implementation of the destructor to perform any additional
	cleanup specific to the derived class.*/

	//Initialize the node parameters with random values
	virtual void init(rne_t& rne) = 0;

	//Forward pass.
	virtual void forward(num_t* inputs) = 0;
	/*
	Good enough for initial testing, but in practice this should be replaced
	with an actual type with a shape defined by data
	to permit additional validation.
	*/

	//Backward pass.
	virtual void reverse(num_t* gradients) = 0;
	/*Expected inputs during the reverse accumulation phase are the loss
	gradients with respect to each output.*/

	//Return the number of (learnable) parameters in the node.
	virtual size_t param_count() const noexcept {
		return 0;
	}
	/*Nodes with no learnable parameters are input & loss nodes.*/

	//Indexing operator for learnable parameters that are altered during training.
	virtual num_t* param(size_t index) {
		return nullptr;
	}
	/*Nodes without learnable parameters should keep this unimplemented.*/

	//Indexing operator for the losss gradients with respect to learnable parameters.
	virtual num_t* gradient(size_t index) {
		return nullptr;
	}
	/*Used by an optimizer to adjust the corresponding parameter and potentially
	for tracking gradient histories (for more sophisitcated optimizers) e.g. AdaGrad*/

	[[nodiscard]] std::string const& name() const noexcept {
		return name_;
	}
	/*The [[nodiscard]] attribute indicates that the caller of the function
	should not ignore the return value. If the return value is ignored,
	the compiler may issue a warning.*/

	//For display of node contents.
	virtual void print() const = 0;

protected:
	friend class Model;

	Model& model_;
	std::string name_;
	std::vector<Node*> antecedents_;
	std::vector<Node*> subsequents_;
};
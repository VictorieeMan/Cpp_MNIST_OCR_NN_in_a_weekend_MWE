#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

// Declaring type aliases
using num_t = float;		//Default precsision, single 32-bit
using rne_t = std::mt19937; //Random number engine, for double 64-bit use std::mt19937_64


enum class Activation {
	ReLU,
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

// Optimizer class
class Optimizer {
public:
	virtual void train(Node& node) = 0;
};

// Model class
class Model {
public:
	Model(std::string name);

	template <typename Node_t, typename... T>
	Node_t& add_node(T&&... args) {
		nodes_.emplace_back(
			std::make_unique<Node_t>(*this, std::forward<T>(args)...)
		);
		return reinterpret_cast<Node_t&>(*nodes_.back());
	}

	void create_edge(Node& dst, Node& src);

	//Initialize the model parameters with provided seed,
	//if seed is 0, use a random seed.
	rne_t::result_type init(rne_t::result_type seed = 0);

	void train(Optimizer& optimizer);

	[[nodiscard]] std::string const& name() const noexcept {
		return name_;
	}

	void print() const;

	//Save and load the model to/from a file.
	void save(std::ofstream& out);
	void load(std::ifstream& in);

private:
	friend class Node;
	
	std::string name_;
	std::vector<std::unique_ptr<Node>> nodes_;
	/*std::unique_ptr is useful for managing objects with dynamic lifetimes, 
	such as objects allocated on the heap with new. It can help prevent memory 
	leaks and make your code safer and easier to read by automatically managing 
	the lifetime of the object.*/
};
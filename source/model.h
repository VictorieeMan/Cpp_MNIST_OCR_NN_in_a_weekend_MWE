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
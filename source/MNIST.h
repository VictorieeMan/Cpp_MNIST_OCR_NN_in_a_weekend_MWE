#pragma once

#include "model.h"
#include <fstream>

class MNIST : public Node {
public:
	//The MNIST data set images have the 28x28 pixel resolution.
	constexpr static size_t DIM = 28 * 28;


};
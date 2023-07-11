// main.cpp : Defines the entry point for the application.
//

#include "CCELossNode.h"
#include "FFNode.h"
#include "GDOptimizer.h"
#include "MNIST.h"
#include "model.h"

#include <iostream>
#include <vector>
#include <cfenv>
#include <cstdio>
#include <cstring>
#include <filesystem>

//Hyperparameters
static constexpr size_t batch_size = 100;

Model create_model(
	std::ifstream& images,
	std::ifstream& labels,
	MNIST** mnist,
	CCELossNode** loss
) {
	//Here we create the model:
	//a simple fully connected network feedforward neural network.
	Model model{ "ff" };

	*mnist = &model.add_node<MNIST>(images, labels);

	FFNode& hidden = model.add_node<FFNode>("hidden", Activation::ReLU, 32, 784);

	FFNode& output = model.add_node<FFNode>("output", Activation::Softmax, 10, 31);

	*loss = &model.add_node<CCELossNode>("loss", 10, batch_size);

	// F.T.R. The structure of our computational graph is completely sequential.
	// In fact, the fully connected node and loss node we've implemented here do
	// not support multiple inputs. Consider adding nodes that support "skip"
	// connections that forward outputs from earlier nodes to downstream nodes
	// that aren't directly adjacent (such skip nodes are used in the ResNet
	// architecture)

	model.create_edge(hidden, **mnist);
	model.create_edge(output, hidden);
	model.create_edge(**loss, output);
	return model;
}

int main(int argc, char* argv[]) {
	std::cout << "Hello user! Pick a mode of operation." << std::endl;
	
	// Creating a vector of arguments;
	// to making switch between command line input and manual input easy
	std::vector<std::string> args(argv, argv + argc);

	if (argc != 2) {
		std::cout << "This program can be launched with arguments like this:" << std::endl;
		std::cout << "Usage: " << argv[0] << " <mode>" << std::endl;
		std::cout << "where <mode> is either \"train\" or \"test\"" << std::endl;
		std::cout << std::endl << "No arguments were provided. Manual mode selected." << std::endl;
		std::cout << "Please enter the mode of operation: ";
		char mode[10];
		std::cin >> mode;

		// Adding to args vector
		args.push_back(mode);
	}

	// Dealing with user commands
	if (args[1] == "train") {
		std::cout << "Training mode" << std::endl;
	}
	else if (args[1] == "test") {
		std::cout << "Testing mode" << std::endl;
	}
	else {
		std::cout << "Unknown mode" << std::endl;
	}
	return 0;
}

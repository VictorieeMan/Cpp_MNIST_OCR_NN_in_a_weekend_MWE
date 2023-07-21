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

std::string filepath_generator(char* argv[], std::string folder, std::string filename) {
	//This generates filepaths relative to the exe file.
	std::string path = argv[0];
	path = path.substr(0, path.find_last_of("\\/") + 1);
	path += folder;
		return path + filename;
}

std::string MNIST_data_filepath(char* argv[], std::string filename) {
	// This function is used to find the path to the MNIST data files.
#if defined(WIN32) || defined(_WIN32) 
  std::string folder = "MNIST_dataset\\";
#else
  std::string folder = "MNIST_dataset/";
#endif
	return filepath_generator(argv, folder, filename);
}

std::ifstream open_file(std::string file_path) {
	// This function is used to open the MNIST data files.
	// It is used to open the files in binary mode.
	std::ifstream file{ file_path, std::ios::binary };
	if (!file) {
		throw std::runtime_error{ "Could not open file " + file_path };
	}
	return file;
}

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
	(*loss)->set_target((*mnist)->label());

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

void train(char* argv[], std::string image_path, std::string label_path) {
	// Uncomment to debug floating point instability in the network
	// feenableexcept(FE_INVALID | FE_OVERFLOW);

	std::printf("Executing training routine\n");

	std::ifstream images = open_file(image_path);
	std::ifstream labels = open_file(label_path);

	MNIST* mnist;
	CCELossNode* loss;
	Model model = create_model(images, labels, &mnist, &loss);

	model.init();

	// The gradient descent optimizer is stateless, but other optimizers may not
	// be. Some optimizers need to track "momentum" or gradient histories.
	// Others may slow the learning rate for each parameter at different rates
	// depending on various factors.
	//
	// F.T.R. Implement an alternative SGDOptimizer that decays the learning
	// rate over time and compare the results against this optimizer that learns
	// at a fixed rate.
	GDOptimizer optimizer{ num_t{0.3} };

	// F.T.R. Here, we've hardcoded the number of batches to train on. In
	// practice, training should halt when the average loss begins to
	// vascillate, indicating that the model is starting to overfit the data.
	// Implement some form of loss-improvement measure to determine when this
	// inflection point occurs and stop accordingly.
	size_t i = 0;
	for (; i != 256; ++i) {
		loss->reset_score();

		for (size_t j = 0; j != batch_size; ++j) {
			mnist->forward();
			loss->reverse();
		}

		model.train(optimizer);
	}

	std::printf("Ran %zu batches (%zu samples each)\n", i, batch_size);

	//Print the average loss computed in the final batch
	loss->print();

	std::ofstream out{
		std::filesystem::current_path() / (model.name() + ".params"),
		std::ios::binary
	};

	model.save(out);
}

void evaluate(char* argv[], std::string image_path, std::string label_path, std::string params_file_path) {
	std::printf("Executing evaluation routine\n");

	std::ifstream images = open_file(image_path);
	std::ifstream labels = open_file(label_path);

	MNIST* mnist;
	CCELossNode* loss;
	// For the data to be loaded properly, the model must be constructed in the
	// same manner as it was constructed during training.
	Model model = create_model(images, labels, &mnist, &loss);

	// Instead of initializing the parameters randomly, here we load it from
	// disk (saved from a previous training run).
	std::ifstream params_file = open_file(params_file_path);
	model.load(params_file);

	// Evaluate all 10000 images in the test set and compute the loss average
	for (size_t i = 0; i != mnist->size(); ++i) {
		mnist->forward();
	}
	mnist->print_last();
	loss->print();
}

int main(int argc, char* argv[]) {
	bool debug = false;
	bool debug_train = false; //if true, debug eval
	if (debug) {
		std::cout << "Debug mode is on" << std::endl;
		std::cout << "Path: " << argv[0] << std::endl;
	}
	std::cout << "Hello user!";

	int loops = 0; //In case of bug, block infty loop
	while (true) {
		std::cout << "Pick a mode of operation." << std::endl;

		// Creating a vector of arguments;
		// to making switch between command line input and manual input easy
		std::vector<std::string> args(argv, argv + argc);

		//Creating filepaths to image & label files
		std::string image_path = MNIST_data_filepath(argv, "train-images.idx3-ubyte");
		std::string label_path = MNIST_data_filepath(argv, "train-labels.idx1-ubyte");
		std::string model_name = "ff.params";
		std::string params_file_path = filepath_generator(argv, "", model_name);


		if (argc != 2 && !debug) {
			std::cout << "This program can be launched with arguments like this:" << std::endl;
			std::cout << "Usage: " << argv[0] << " <mode>" << std::endl;
			std::cout << "where <mode> is either \"train\" or \"eval\"" << std::endl;
			std::cout << std::endl << "No arguments were provided. Manual mode selected." << std::endl;
			std::cout << "Enter q to quit." << std::endl;
			std::cout << "Please enter the mode of operation: ";
			char mode[10];
			std::cin >> mode;
			std::cout << std::endl;

			if (mode[0] == 'q') {
				std::cout << "Exiting program." << std::endl;
				break;
			}

			// Adding to args vector
			args.push_back(mode);
		}
		else if (argc != 2) {
			if (debug_train) {
				//For debugging purposes, faster then typing in the command line
				args.push_back("train");
			}
			else {
				args.push_back("eval");
			}
		}

		// Dealing with user commands
		if (args[1] == "train") {
			std::cout << "Training mode" << std::endl;
			train(argv, image_path, label_path);
		}
		else if (args[1] == "eval") {
			std::cout << "Evaluation mode" << std::endl;
			evaluate(argv, image_path, label_path, params_file_path);
		}
		else {
			std::cout << "Unknown mode in input." << std::endl;
		}

		if (argc > 1) {
			//Don't loop on cli input
			break;
		}
		else {
			++loops;
			if (loops > 1000) {
				std::cout << "Infinite loop detected. Exiting." << std::endl;
				break;
			}
		}
	}

	std::cout << "Program finished." << std::endl;

	if (debug == false) {
		// To keep the window open until keys are pressed.
		int dummy = 0;
		std::cin >> dummy;
	}

	return 0;
}

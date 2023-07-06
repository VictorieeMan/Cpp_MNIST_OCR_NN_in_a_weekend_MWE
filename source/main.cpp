// main.cpp : Defines the entry point for the application.
//

#include <iostream>
#include <vector>

//using namespace std;

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

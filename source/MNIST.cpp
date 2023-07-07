#include "MNIST.h"

#include <cstdio>
#include <stdexcept>

//Reads the MNIST data from the files.
//Read 4 bytes from the file and return it as an unsigned int.
void read_be(std::ifstream& in, uint32_t* out) {
	char* buf = reinterpret_cast<char*>(out);
	in.read(buf, 4);

	std::swap(buf[0], buf[3]);
	std::swap(buf[1], buf[2]);
}
/*The reinterpret_cast operator performs a low-level, bit-wise conversion
between pointer types. It’s typically used in situations where you need to
treat a block of memory as if it were a different type, or when you need to
perform a type conversion that would otherwise be disallowed by the compiler.*/


//Constructs the MNIST input node;
//Loads the MNIST data from the files.
//And checks for well formatted MNIST data.
MNIST::MNIST(Model& model, std::ifstream& images, std::ifstream& labels)
	: Node{ model, "MNIST input" }
	, images_{ images }
	, labels_{ labels }
{	
	//Check for well formatted MNIST data.
	uint32_t image_magic;
	read_be(images, &image_magic);
	if (image_magic != 2051) {
		throw std::runtime_error{"Images file is malformed"};
	}
	read_be(images, &image_count_);

	uint32_t labels_magic;
	read_be(labels, &labels_magic);
	if (labels_magic != 2049) {
		throw std::runtime_error{"Labels file appears to be malformed"};
	}
	/*Exception checks dependent on specific knowledge of the MNIST data set.*/

	uint32_t label_count;
	read_be(labels, &label_count);
	if (label_count != image_count_) {
		throw std::runtime_error{
			"Label count does not match image count."
		};
	}
	/*The number of labels and images must be the same.*/

	uint32_t rows;
	uint32_t columns;
	read_be(images, &rows);
	read_be(images, &columns);
	if (rows != 28 || columns != 28) {
		throw std::runtime_error{
			"Images are not 28x28 pixels in size, as expected."
		};
	}

	printf("Successfully loaded %u images.\n", image_count_);
}
#include "MNIST.h"

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
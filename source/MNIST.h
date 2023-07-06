//#pragma once

#include "model.h"
#include <fstream>

class MNIST : public Node {
public:
	//The MNIST data set images have the 28x28 pixel resolution.
	constexpr static size_t DIM = 28 * 28;

	//Data input, of images and labels.
	MNIST(Model& model, std::ifstream images, std::ifstream& labels);

	void init(rne_t&) override{}

	//Since MNIST is the input node, the arguments to this function is ignored.
	void forward(num_t* data = nullptr) override;

	//For the input node there are no parameters to update.
	void reverse(num_t* data = nullptr) override;

	//For parsing the next image and label into memory.
	void read_next();

	void print() const override;

	[[nodiscard]] size_t size() const noexcept {
		return image_count_;
	}

	[[nodiscard]] num_t const* data() const noexcept {
		return data_;
	}

	[[nodiscard]] num_t* data() noexcept {
		return data_;
	}

	[[nodiscard]] num_t* label() noexcept {
		return label_;
	}

	[[nodiscard]] num_t const* label() const noexcept	{
		return label_;
	}

	// Quick ASCII visualization of the last read image. For best results,
	// ensure that your terminal font is a monospace font
	void print_last();

private:
	std::ifstream& images_;
	std::ifstream& labels_;
	uint32_t image_count_;
	/*Data from the image file is read as one - byte unsigned values,
	later converted to num_t.*/

	char buf_[DIM];
	// All images are resized (with antialiasing) to a 28 x 28 row-major raster
	num_t data_[DIM];
	num_t label_[10];
};
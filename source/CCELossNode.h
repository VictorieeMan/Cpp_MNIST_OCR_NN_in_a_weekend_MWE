#pragma once

#include "model.h"

// Categorical Cross-Entropy Loss Node
// Assumes input data is "one-hot encoded," with size equal to the number of
// possible classifications, where the "answer" has a single "1" (aka hot value)
// in one of the classification positions and zero everywhere else.

class CCELossNode : public Node {
public:
	CCELossNode(
		Model& model,
		std::string name,
		uint16_t input_size,
		size_t batch_size
	);

	//This node doesn't need initialization
	void init(rne_t&) override {}

	void forward(num_t* inputs) override;
	//As a loss node, we ignore the arguments to this method
	//(THe gradient of the loss with respect to itself is unity)

	void reverse(num_t* gradients = nullptr) override;

	void print() const override;

	void set_target(num_t const* target) {
		target_ = target;
	}

	num_t accuracy() const;
	num_t avg_loss() const;
	void reset_score();

private:
	uint16_t input_size_;

	//To make loss independet from batch size, we divide by batch size.
	//And hence minimize the average loss instead of net losses,
	//to help keep training paremeters constant across different batch sizes.
	num_t inv_batch_size_;
	num_t loss_;
	num_t const* target_;
	num_t* last_input_;
	// Stores the last active classification in the target one-hot encoding
	size_t active_;
	num_t cumulative_loss_{ 0.0 };
	// Stores the number of correct & incorrect classifications
	size_t correct_ = 0;
	size_t incorrect_ = 0;
	std::vector<num_t> gradients_;
};
#pragma once

#include "model.h"

// Categorical Cross-Entropy Loss Node
// Assumes input data is "one-hot encoded," with size equal to the number of
// possible classifications, where the "answer" has a single "1" (aka hot value)
// in one of the classification positions and zero everywhere else.

class CCELossNode : public Node {
public:
	CCELossNode(
		std::string name,
		uint16_t input_size,
		size_t batch_size
	);

	//This node doesn't need initialization
	void init(rne_t&) override {}

	void forward(num_t* inputs) override;
	//As a loss node, we ignore the arguments to this method
	//(THe gradient of the loss with respect to itself is unity)

	void reverse(num_t* gradients) override;

	void print() const override;

	void set_target(num_t const* target) {
		target_ = target;
	}

	num_t accuracy() const;
	num_t avg_loss() const;
	void reset_score();

private:
	num_t const* target_;
};
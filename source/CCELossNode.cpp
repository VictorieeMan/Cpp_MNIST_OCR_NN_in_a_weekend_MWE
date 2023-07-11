#include "CCELossNode.h"

#include <limits>

/*
The CCELossNode is similar to other nodes in that it implements a forward
pass for computing the loss of a given sample, and a reverse pass to compute
gradients of that loss and pass them back to the antecedent node. Distinct from
the previous nodes is that the argument to CCELossNode::reverse is ignored
as the loss node is not expected to have any subsequents.
*/

CCELossNode::CCELossNode(
	Model& model,
	std::string name,
	uint16_t input_size,
	size_t batch_size
)
	: Node{model, std::move(name)}
	, input_size_{input_size}
	, inv_batch_size_{ num_t{1.0} / static_cast<num_t>(batch_size) }
{
	//When we deliver a gradient back, we deliver just the loss gradient with
	//respect to any input and the index that was "hot" in the second argument.
	gradients_.resize(input_size_);
}

void CCELossNode::forward(num_t* data) {
	//The cross-entropy categorical loss is defined as -\sum_i(q_i * log(p_i))
	//where p_i is the predicted probability and q_i the expected probability.
	//
	//In information theory, by convention, lim_{x approaches 0}(x log(x)) = 0.

	num_t max{ 0.0 };
	size_t max_index;

	loss_ = num_t{ 0.0 };
	for (size_t i = 0; i != input_size_; ++i) {
		if (data[i] > max) {
			max_index = i;
			max = data[i];
		}

		// Because the target vector is one-hot encoded, most of these terms
		// will be zero, but we leave the full calculation here to be explicit
		// and in the event we want to compute losses against probability
		// distributions that arent one-hot. In practice, a faster code path
		// should be employed if the targets are known to be one-hot
		// distributions.
		/*One-hot encoding is a technique used to represent categorical data as 
		numerical data that can be used in machine learning models. It involves 
		representing each category as a binary vector with a single “1” in the 
		position corresponding to the category, and "0"s everywhere else.*/

		loss_ -= target_[i] * std::log(
			//Prevent, undefined results when the prediction is zero.
			std::max(data[i], std::numeric_limits<num_t>::epsilon())
		);

		if (target_[i] != num_t{0.0}) {
			active_ = i;
		}

		// NOTE: The astute reader may notice that the gradients associated with
		// many of the loss node's input signals will be zero because the
		// cross-entropy is performed with respect to a one-hot vector.
		// Fortunately, because the layer preceding the output layer is a
		// softmax layer, the gradient from the single term contributing in the
		// above expression has a dependency on *every* softmax output unit (all
		// outputs show up in the summation in the softmax denominator).
	}

	//Accounting
	if (max_index == active_) {
		++correct_;
	}
	else {
		++incorrect_;
	}

	cumulative_loss_ += loss_;


	//Store the data pointer to compute gradients later
	last_input_ = data;
}
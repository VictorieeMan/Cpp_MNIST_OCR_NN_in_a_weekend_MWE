#include "CCELossNode.h"

#include <limits>

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
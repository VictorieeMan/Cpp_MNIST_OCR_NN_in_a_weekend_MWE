#include "GDOptimizer.h"
#include "model.h"

#include <cmath>

namespace
{

std::vector<float> get_gradient(Node& node)
{
  std::vector<float> gradient;
	for (size_t i = 0; i < node.param_count(); ++i) {
    gradient.push_back(*node.gradient(i));
	}
  return gradient;
}

num_t get_l2_norm_squared(const std::vector<num_t>& gradient)
{
  num_t l2_squared = 0;
  for (const num_t g : gradient) {
    l2_squared += g * g;
  }
  return l2_squared;
}

}

GDOptimizer::GDOptimizer(num_t eta)
	: eta_{ eta }
{

}

void GDOptimizer::train(Node& node) {
  // Gradient clipping (/ rescaling).
  auto gradient = get_gradient(node);
  num_t l2_squared = get_l2_norm_squared(gradient);
  if(l2_squared > 100)
  {
    num_t l2_norm = 1.0/std::sqrt(l2_squared);
    for (size_t i = 0; i < node.param_count(); ++i)
    {
      gradient[i] *= l2_norm;
    }
  }
  /*Gradient clipping is a method where the error derivative is changed or
  clipped to a threshold during backward propagation through the network,
  and using the clipped gradients to update the weights*/

  // Update parameters using the gradient and learning rate eta.
	size_t param_count = node.param_count();
	for (size_t i = 0; i < gradient.size(); ++i)
  {
		num_t& param = *node.param(i);

		param = param - eta_ * gradient[i];

		// Reset the gradient which will be accumulated again in the next
		// training epoch
		*node.gradient(i) = num_t{ 0.0 };
	}
}
#pragma once

#include "model.h"

// Categorical Cross-Entropy Loss Node
// Assumes input data is "one-hot encoded," with size equal to the number of
// possible classifications, where the "answer" has a single "1" (aka hot value)
// in one of the classification positions and zero everywhere else.

class CCELossNode : public Node {
public:

private:
};
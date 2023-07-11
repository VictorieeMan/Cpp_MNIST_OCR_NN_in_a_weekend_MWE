#pragma once

#include "model.h"

// Note that this class defines the general gradient descent algorithm. It can
// be used as part of the *Stochastic* gradient descent algorithm (aka SGD) by
// invoking it after smaller batches of training data are evaluated.
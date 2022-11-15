#pragma once
#include <vector>

#include "../../Core.h"

DRAGON_BEGIN

DRAGON_API double maxPool(const std::vector<double>& values);

DRAGON_API std::vector<double> maxPoolDiff(const std::vector<double>& values, double value);

DRAGON_END
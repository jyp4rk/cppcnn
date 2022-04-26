// Copyright 2022 Jaiyoung Park
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <algorithm>
#include <iostream>

#include "CIFAR.h"
#include "Layer.h"
#include "Layer/AvePooling.h"
#include "Layer/Conv.h"
#include "Layer/FullyConnected.h"
#include "Layer/MaxPooling.h"
#include "Layer/ReLU.h"
#include "Layer/Sigmoid.h"
#include "Layer/Softmax.h"
#include "Layer/Square.h"
#include "MNIST.h"
#include "Network.h"
#include "cnpy.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>
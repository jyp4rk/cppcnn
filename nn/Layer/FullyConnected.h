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

#ifndef NN_LAYER_FULLY_CONNECTED_H_
#define NN_LAYER_FULLY_CONNECTED_H_

#include "Layer.h"
#include <vector>

class FullyConnected : public Layer
{
private:
  const int dim_in;
  const int dim_out;

  void init();

public:
  Matrix weight;      // weight parameter
  Vector bias;        // bias paramter
  Matrix grad_weight; // gradient w.r.t weight
  Vector grad_bias;   // gradient w.r.t bias
  FullyConnected(const int dim_in, const int dim_out) : dim_in(dim_in), dim_out(dim_out)
  {
    init();
  }

  void forward(const Matrix &bottom);
  int output_dim() { return dim_out; }
  std::vector<long double> get_parameters() const;
  Matrix get_weight() const;
  Vector get_bias() const;
  std::vector<long double> get_derivatives() const;
  void set_parameters(const std::vector<long double> &param);
};

#endif // NN_LAYER_FULLY_CONNECTED_H_

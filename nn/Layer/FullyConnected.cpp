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

#include "FullyConnected.h"

void FullyConnected::init()
{
  weight.resize(dim_in, dim_out);
  bias.resize(dim_out);
  grad_weight.resize(dim_in, dim_out);
  grad_bias.resize(dim_out);
  set_normal_random(weight.data(), weight.size(), 0, 0.01);
  set_normal_random(bias.data(), bias.size(), 0, 0.01);
}

void FullyConnected::forward(const Matrix &bottom)
{
  // z = w' * x + b
  const int n_sample = bottom.cols();
  top.resize(dim_out, n_sample);
  top = weight.transpose() * bottom;
  top.colwise() += bias;
}

std::vector<long double> FullyConnected::get_parameters() const
{
  std::vector<long double> res(weight.size() + bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(weight.data(), weight.data() + weight.size(), res.begin());
  std::copy(bias.data(), bias.data() + bias.size(),
            res.begin() + weight.size());
  return res;
}
Matrix FullyConnected::get_weight() const
{
  return weight;
}
Vector FullyConnected::get_bias() const
{
  return bias;
}

void FullyConnected::set_parameters(const std::vector<long double> &param)
{
  if (static_cast<int>(param.size()) != weight.size() + bias.size())
  {
    std::cerr << "weight size : " << weight.size() << "\tbias size : " << bias.size() << std::endl;
    throw std::invalid_argument("Parameter size does not match");
  }
  std::copy(param.begin(), param.begin() + weight.size(), weight.data());
  std::copy(param.begin() + weight.size(), param.end(), bias.data());
}

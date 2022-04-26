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

#include "Network.h"

void Network::forward(const Matrix &input)
{
  if (layers.empty())
    return;
  layers[0]->forward(input);
  for (int i = 1; i < layers.size(); i++)
  {
    layers[i]->forward(layers[i - 1]->output());
  }
}

std::vector<std::vector<long double>> Network::get_parameters() const
{
  const int n_layer = layers.size();
  std::vector<std::vector<long double>> res;
  res.reserve(n_layer);
  for (int i = 0; i < n_layer; i++)
  {
    res.push_back(layers[i]->get_parameters());
  }
  return res;
}

void Network::set_parameters(
    const std::vector<std::vector<long double>> &param)
{
  const int n_layer = layers.size();
  if (static_cast<int>(param.size()) != n_layer)
  {
    std::cerr << "param size : " << param.size() << "\tn_layer : " << n_layer
              << std::endl;
    throw std::invalid_argument("Parameter size does not match");
  }
  for (int i = 0; i < n_layer; i++)
  {
    layers[i]->set_parameters(param[i]);
  }
}

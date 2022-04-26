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

#include "Sigmoid.h"

void Sigmoid::forward(const Matrix &bottom)
{
  // a = 1 / (1 + exp(-z))
  top.array() = 1.0 / (1.0 + (-bottom).array().exp());
}

void Sigmoid::backward(const Matrix &bottom, const Matrix &grad_top)
{
  // d(L)/d(z_i) = d(L)/d(a_i) * d(a_i)/d(z_i)
  // d(a_i)/d(z_i) = a_i * (1-a_i)
  Matrix da_dz = top.array().cwiseProduct(1.0 - top.array());
  grad_bottom = grad_top.cwiseProduct(da_dz);
}

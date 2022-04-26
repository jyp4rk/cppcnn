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

#include "Softmax.h"

void Softmax::forward(const Matrix &bottom)
{
  // a = exp(z) / \sum{ exp(z) }
  top.array() = (bottom.rowwise() - bottom.colwise().maxCoeff()).array().exp();
  RowVector z_exp_sum = top.colwise().sum(); // \sum{ exp(z) }
  top.array().rowwise() /= z_exp_sum;
}

void Softmax::backward(const Matrix &bottom, const Matrix &grad_top)
{
  // d(L)/d(z_i) = \sum{ d(L)/d(a_j) * d(a_j)/d(z_i) }
  // = \sum_(i!=j){ d(L)/d(a_j) * d(a_j)/d(z_i) } + d(L)/d(a_i) * d(a_i)/d(z_i)
  // = a_i * ( d(L)/d(a_i) - \sum{a_j * d(L)/d(a_j)} )
  RowVector temp_sum = top.cwiseProduct(grad_top).colwise().sum();
  grad_bottom.array() = top.array().cwiseProduct(grad_top.array().rowwise() - temp_sum);
}

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

#ifndef NN_LAYER_H_
#define NN_LAYER_H_

#define EIGEN_NO_CUDA
#include "utils.h"
#include <Eigen/Core>
#include <vector>

class Layer
{
protected:
  Matrix top;         // layer output
  Matrix grad_bottom; // gradient w.r.t input

public:
  virtual ~Layer() {}

  virtual void forward(const Matrix &bottom) = 0;
  virtual const Matrix &output() { return top; }
  virtual int output_dim() { return -1; }
  virtual std::vector<long double> get_parameters() const
  {
    return std::vector<long double>();
  }
  virtual Matrix get_weight() const { return Matrix(0, 0); }
  virtual Vector get_bias() const { return Vector(0); }
  virtual void set_parameters(const std::vector<long double> &param) {}
};

#endif // NN_LAYER_H_

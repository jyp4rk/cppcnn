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

#ifndef SRC_LAYER_CONV_H_
#define SRC_LAYER_CONV_H_

#include "Layer.h"
#include <vector>

class Conv : public Layer
{
private:
  const int dim_in;
  int dim_out;

  int channel_in;
  int height_in;
  int width_in;
  int channel_out;
  int height_kernel;
  int width_kernel;
  int stride;
  int pad_h;
  int pad_w;

  int height_out;
  int width_out;

  //  Matrix grad_weight;  // gradient w.r.t weight
  //  Vector grad_bias;  // gradient w.r.t bias

  std::vector<Matrix> data_cols;

  void init();

public:
  Matrix weight; // weight param, size=channel_in*h_kernel*w_kernel*channel_out
  Vector bias;   // bias param, size = channel_out
  Conv(int channel_in, int height_in, int width_in, int channel_out,
       int height_kernel, int width_kernel, int stride = 1, int pad_w = 0,
       int pad_h = 0) : dim_in(channel_in * height_in * width_in),
                        channel_in(channel_in), height_in(height_in), width_in(width_in),
                        channel_out(channel_out), height_kernel(height_kernel),
                        width_kernel(width_kernel), stride(stride), pad_w(pad_w), pad_h(pad_h)
  {
    init();
  }

  void forward(const Matrix &bottom);
  void im2col(const Vector &image, Matrix &data_col);
  void col2im(const Matrix &data_col, Vector &image);
  int output_dim() { return dim_out; }
  std::vector<long double> get_parameters() const;
  Matrix get_weight() const;
  Vector get_bias() const;
  std::vector<long double> get_derivatives() const;
  void set_parameters(const std::vector<long double> &param);
};

#endif // SRC_LAYER_CONV_H_
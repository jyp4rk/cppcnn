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

#include "Conv.h"
#include <iostream>
#include <math.h>

void Conv::init()
{
  height_out = (1 + (height_in - height_kernel + 2 * pad_h) / stride);
  width_out = (1 + (width_in - width_kernel + 2 * pad_w) / stride);
  dim_out = height_out * width_out * channel_out;

  weight.resize(channel_in * height_kernel * width_kernel, channel_out);
  bias.resize(channel_out);
  set_normal_random(weight.data(), weight.size(), 0, 0.01);
  set_normal_random(bias.data(), bias.size(), 0, 0.01);
  // std::cout << weight.colwise().sum() << std::endl;
  // std::cout << weight.colwise().sum() + bias.transpose() << std::endl;
}

// im2col, used for bottom
// image size: Vector (height_in * width_in * channel_in)
// data_col size: Matrix (hw_out, hw_kernel * channel_in)
void Conv::im2col(const Vector &image, Matrix &data_col)
{
  int hw_in = height_in * width_in;
  int hw_kernel = height_kernel * width_kernel;
  int hw_out = height_out * width_out;
  // im2col
  data_col.resize(hw_out, hw_kernel * channel_in);
  for (int c = 0; c < channel_in; c++)
  {
    Vector map = image.block(hw_in * c, 0, hw_in, 1); // c-th channel map
    for (int i = 0; i < hw_out; i++)
    {
      int step_h = i / width_out;
      int step_w = i % width_out;
      int start_idx = step_h * width_in * stride + step_w * stride; // left-top idx of window
      for (int j = 0; j < hw_kernel; j++)
      {
        int cur_col = start_idx % width_in + j % width_kernel - pad_w; // col after padding
        int cur_row = start_idx / width_in + j / width_kernel - pad_h;
        if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
            cur_row >= height_in)
        {
          data_col(i, c * hw_kernel + j) = 0;
        }
        else
        {
          // int pick_idx = start_idx + (j / width_kernel) * width_in + j % width_kernel;
          int pick_idx = cur_row * width_in + cur_col;
          data_col(i, c * hw_kernel + j) = map(pick_idx); // pick which pixel
        }
      }
    }
  }
}

void Conv::forward(const Matrix &bottom)
{
  int n_sample = bottom.cols();
  top.resize(height_out * width_out * channel_out, n_sample);
  data_cols.resize(n_sample);
  for (int i = 0; i < n_sample; i++)
  {
    // im2col
    Matrix data_col;
    im2col(bottom.col(i), data_col);
    data_cols[i] = data_col;
    // conv by product
    Matrix result = data_col * weight; // result: (hw_out, channel_out)
    result.rowwise() += bias.transpose();
    top.col(i) = Eigen::Map<Vector>(result.data(), result.size());
  }
}

// col2im, used for grad_bottom
// data_col size: Matrix (hw_out, hw_kernel * channel_in)
// image size: Vector (height_in * width_in * channel_in)
void Conv::col2im(const Matrix &data_col, Vector &image)
{
  int hw_in = height_in * width_in;
  int hw_kernel = height_kernel * width_kernel;
  int hw_out = height_out * width_out;
  // col2im
  image.resize(hw_in * channel_in);
  image.setZero();
  for (int c = 0; c < channel_in; c++)
  {
    for (int i = 0; i < hw_out; i++)
    {
      int step_h = i / width_out;
      int step_w = i % width_out;
      int start_idx = step_h * width_in * stride + step_w * stride; // left-top idx of window
      for (int j = 0; j < hw_kernel; j++)
      {
        int cur_col = start_idx % width_in + j % width_kernel - pad_w; // col after padding
        int cur_row = start_idx / width_in + j / width_kernel - pad_h;
        if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
            cur_row >= height_in)
        {
          continue;
        }
        else
        {
          // int pick_idx = start_idx + (j / width_kernel) * width_in + j % width_kernel;
          int pick_idx = cur_row * width_in + cur_col;
          image(c * hw_in + pick_idx) += data_col(i, c * hw_kernel + j); // pick which pixel
        }
      }
    }
  }
}

std::vector<long double> Conv::get_parameters() const
{
  std::vector<long double> res(weight.size() + bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(weight.data(), weight.data() + weight.size(), res.begin());
  std::copy(bias.data(), bias.data() + bias.size(), res.begin() + weight.size());
  return res;
}
Matrix Conv::get_weight() const
{
  return weight;
}
Vector Conv::get_bias() const
{
  return bias;
}
void Conv::set_parameters(const std::vector<long double> &param)
{
  if (static_cast<int>(param.size()) != weight.size() + bias.size())
  {
    std::cerr << "weight size : " << weight.size() << "\tbias size : " << bias.size() << std::endl;
    throw std::invalid_argument("Parameter size does not match");
  }
  std::copy(param.begin(), param.begin() + weight.size(), weight.data());
  std::copy(param.begin() + weight.size(), param.end(), bias.data());
}

// void Conv::set_parameters(const Matrix& param) {
//   if(static_cast<int>(param.size()) != weight.size() + bias.size()){
//			std:: cerr << "weight size : " << weight.size() << "\tbias size : " << bias.size() << std::endl;
//       throw std::invalid_argument("Parameter size does not match");
//	}
//   std::copy(param.begin(), param.begin() + weight.size(), weight.data());
//   std::copy(param.begin() + weight.size(), param.end(), bias.data());
// }

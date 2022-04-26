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

#include "MaxPooling.h"
#include <iostream>
#include <limits>
#include <math.h>

void MaxPooling::init()
{
  channel_out = channel_in;
  height_out = (1 + ceil((height_in - height_pool) * 1.0 / stride));
  width_out = (1 + ceil((width_in - height_pool) * 1.0 / stride));
  dim_out = height_out * width_out * channel_out;
}

void MaxPooling::forward(const Matrix &bottom)
{
  int n_sample = bottom.cols();
  int hw_in = height_in * width_in;
  int hw_pool = height_pool * width_pool;
  int hw_out = height_out * width_out;
  top.resize(dim_out, n_sample);
  top.setZero();
  top.array() += std::numeric_limits<long double>::lowest();
  max_idxs.resize(n_sample, std::vector<int>(dim_out, 0));
  for (int i = 0; i < n_sample; i++)
  {
    Vector image = bottom.col(i);
    for (int c = 0; c < channel_in; c++)
    {
      for (int i_out = 0; i_out < hw_out; i_out++)
      {
        int step_h = i_out / width_out;
        int step_w = i_out % width_out;
        // left-top idx of window in raw image
        int start_idx = step_h * width_in * stride + step_w * stride;
        for (int i_pool = 0; i_pool < hw_pool; i_pool++)
        {
          if (start_idx % width_in + i_pool % width_pool >= width_in ||
              start_idx / width_in + i_pool / width_pool >= height_in)
          {
            continue; // out of range
          }
          int pick_idx = start_idx + (i_pool / width_pool) * width_in + i_pool % width_pool + c * hw_in;
          if (image(pick_idx) >= top(c * hw_out + i_out, i))
          { // max pooling
            top(c * hw_out + i_out, i) = image(pick_idx);
            max_idxs[i][c * hw_out + i_out] = pick_idx;
          }
        }
      }
    }
  }
}

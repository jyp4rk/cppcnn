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

#ifndef SRC_LAYER_MAX_POOLING_H_
#define SRC_LAYER_MAX_POOLING_H_

#include "Layer.h"
#include <vector>

class MaxPooling : public Layer
{
private:
  int channel_in;
  int height_in;
  int width_in;
  int dim_in;

  int height_pool; // pooling kernel height
  int width_pool;  // pooling kernel width
  int stride;      // pooling stride

  int channel_out;
  int height_out;
  int width_out;
  int dim_out;

  std::vector<std::vector<int>> max_idxs; // index of max values

  void init();

public:
  MaxPooling(int channel_in, int height_in, int width_in,
             int height_pool, int width_pool, int stride = 1) : dim_in(channel_in * height_in * width_in),
                                                                channel_in(channel_in), height_in(height_in), width_in(width_in),
                                                                height_pool(height_pool), width_pool(width_pool), stride(stride)
  {
    init();
  }

  void forward(const Matrix &bottom);
  int output_dim() { return dim_out; }
};

#endif // SRC_LAYER_MAX_POOLING_H_

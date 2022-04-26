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

#include "CIFAR.h"

void CIFAR::read()
{
  read_cifar_data(data_dir + "cifar-10-batches-bin/" + "data_batch_1.bin",
                  train_data, train_labels);
  read_cifar_data(data_dir + "cifar-10-batches-bin/" + "test_batch.bin",
                  test_data, test_labels);
}
void CIFAR::transform(long double mean, float stdev)
{
  train_data = (train_data -
                Matrix::Constant(train_data.rows(), train_data.cols(), mean)) /
               stdev;
  test_data =
      (test_data - Matrix::Constant(test_data.rows(), test_data.cols(), mean)) /
      stdev;
}

void CIFAR::read_cifar_data(std::string filename, Matrix &data,
                            Matrix &labels)
{
  std::ifstream file(filename, std::ios::binary);
  if (file.is_open())
  {
    int number_of_images = 1;
    int n_channels = 3;
    int n_rows = 32;
    int n_cols = 32;
    data.resize(n_cols * n_rows * n_channels, number_of_images);
    labels.resize(number_of_images, 1);
    for (int i = 0; i < number_of_images; i++)
    {
      unsigned char label = 0;
      file.read((char *)&label, sizeof(label));
      labels(i) = (long double)label;
      for (int ch = 0; ch < n_channels; ch++)
      {
        for (int r = 0; r < n_rows; r++)
        {
          for (int c = 0; c < n_cols; c++)
          {
            unsigned char image = 0;
            file.read((char *)&image, sizeof(image));
            data(ch * n_rows * n_cols + r * n_cols + c, i) = (long double)image;
          }
        }
      }
    }
  }
}
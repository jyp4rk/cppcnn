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

#ifndef NN_MNIST_H_
#define NN_MNIST_H_

#include "cnpy.h"
#include "utils.h"
#include <fstream>
#include <iostream>
#include <string>

class MNIST
{
private:
  std::string data_dir;

public:
  Matrix train_data;
  Matrix train_labels;
  Matrix test_data;
  Matrix test_labels;

  void read_mnist_data(std::string filename, Matrix &data);
  void read_mnist_label(std::string filename, Matrix &labels);
  void transform(long double mean, float stdev);

  explicit MNIST(std::string data_dir) : data_dir(data_dir) {}
  void read();
};
std::vector<long double> loadParameter(std::string moduleName,
                                       std::string fileName);
std::vector<long double> loadParameter();
std::vector<long double> genParameter(int channel_in, int height_in, int width_in,
                                      int channel_out, int height_kernel, int width_kernel);

#endif

/* FIXME CSV version
void MNIST::read_mnist_data(std::string filename, Matrix& labelset, Matrix&
dataset){ std::string matrixString; std::string rowString; std::string
matrixEntry;

        std::ifstream dataFile(filename);
        //size of the matrix : 784 = 28 x 28
        const int matrixSize = 784;
  //number of data in dataset, should subtract 1 due to .npy characteristics
  const int numMatrix = std::count(std::istreambuf_iterator<char>(dataFile),
std::istreambuf_iterator<char>(),'\n') -1; dataset.resize(matrixSize,numMatrix);
//variable for dataset labelset.resize(1,numMatrix);   //variable for dataset

  //dataset.resize(matrixSize,1000);   //variable for dataset
  //labelset.resize(1,1000);   //variable for dataset
  dataFile.seekg(0);                      //reset istream_buf_iterator to 0

  getline(dataFile,matrixString);
  std::string label;                      //label

  for(int itr = 0; itr <dataset.cols(); itr++)
  {
    getline(dataFile,matrixString);
    Matrix matrix(matrixSize,1);       //matrix
    std::stringstream matrixStringstream(matrixString);
    getline(matrixStringstream,label,',');
    for(int i = 0 ; i<matrixSize; i++)
    {
      getline(matrixStringstream,matrixEntry, ',');
      matrix(i,0) = stof(matrixEntry); //TODO : a pixel encoded to long doub
    }
    labelset.col(itr) <<stof(label);
    dataset.col(itr)<<matrix;
  }
  std::cout <<"num data" << dataset.cols() << std::endl;
}
*/

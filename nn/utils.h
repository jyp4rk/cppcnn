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

#ifndef NN_UTILS_H_
#define NN_UTILS_H_
#define EIGEN_NO_CUDA
#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

typedef Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Matrix<long double, Eigen::Dynamic, 1> Vector;
typedef Eigen::Array<long double, 1, Eigen::Dynamic> RowVector;

static std::default_random_engine generator;

// Normal distribution: N(mu, sigma^2)
inline void set_normal_random(long double *arr, int n, long double mu,
                              long double sigma)
{
  std::normal_distribution<long double> distribution(mu, sigma);
  for (int i = 0; i < n; i++)
  {
    arr[i] = distribution(generator);
  }
}

// shuffle cols of matrix
inline void shuffle_data(Matrix &data, Matrix &labels)
{
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(data.cols());
  perm.setIdentity();
  std::random_shuffle(perm.indices().data(),
                      perm.indices().data() + perm.indices().size());
  data = data * perm; // permute columns
  labels = labels * perm;
}

// encode discrete values to one-hot values
inline Matrix one_hot_encode(const Matrix &y, int n_value)
{
  int n = y.cols();
  Matrix y_onehot = Matrix::Zero(n_value, n);
  for (int i = 0; i < n; i++)
  {
    y_onehot(int(y(i)), i) = 1;
  }
  return y_onehot;
}

// classification accuracy
inline long double compute_accuracy(const Matrix &preditions,
                                    const Matrix &labels)
{
  int n = preditions.cols();
  long double acc = 0;
  for (int i = 0; i < n; i++)
  {
    Matrix::Index max_index;
    long double max_value = preditions.col(i).maxCoeff(&max_index);
    // std::cout << "max_index : " << max_index << std::endl;
    acc += int(max_index) == labels(i);
  }
  return acc / n;
}

#endif // SRC_UTILS_H_

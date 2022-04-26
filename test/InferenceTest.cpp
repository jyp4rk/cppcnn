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
//! InferenceTest for MNIST

#include "cnpy.h"
#include "common.h"
#include "gtest/gtest.h"
#include <chrono>

namespace cppnn
{
    using namespace std;

    //! Prototype of NN layers
    //! Conv(int channel_in, int height_in, int width_in, int channel_out,
    //     int height_kernel, int width_kernel, int stride = 1, int pad_w = 0,int
    //     pad_h = 0) : //! MaxPooling(int channel_in, int height_in, int width_in,
    //     //           int height_pool, int width_pool, int stride = 1)

    class InferenceTest : public ::testing::Test
    {
    protected:
        Matrix dmat;
        Network dnn;
        MNIST *data;
        shared_ptr<Layer> conv1;
        shared_ptr<Layer> fc1;
        shared_ptr<Layer> fc2;
        Matrix conv1_output;
        Matrix fc1_output;
        Matrix fc2_output;
        Matrix output;

        std::vector<std::vector<long double>> param;

        virtual void SetUp()
        {
            data = new MNIST("../mnist_data/");
            data->read();
            conv1 = make_shared<Conv>(1, 28, 28, 1, 5, 5, 2);
            fc1 = make_shared<FullyConnected>(144, 40);
            fc2 = make_shared<FullyConnected>(40, 10);

            dnn.add_layer(conv1);
            dnn.add_layer(fc1);
            dnn.add_layer(fc2);

            param.push_back(loadParameter("nn1", "conv1"));
            param.push_back(loadParameter("nn1", "fc1"));
            param.push_back(loadParameter("nn1", "fc2"));
            dnn.set_parameters(param);
            data->transform(0, 256);
            data->transform(0.1307, 0.3018);
            dnn.forward(data->test_data);
            output = dnn.output();
            conv1_output = dnn.layers[0]->output();
            fc1_output = dnn.layers[1]->output();
            fc2_output = dnn.layers[2]->output();
        }
        virtual void TearDown() {}
    };
    TEST_F(InferenceTest, Conv1_layer)
    {
        // Place test code here
    }
    TEST_F(InferenceTest, FC1_layer)
    {
        // Place test code here
    }
    TEST_F(InferenceTest, FC2_layer)
    {
        // Place test code here
    }
    TEST_F(InferenceTest, NetworkInference)
    {
        // Place test code here
    }

} // namespace cppnn
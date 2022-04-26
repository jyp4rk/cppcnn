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

#include "common.h"

namespace cppnn
{
    using namespace std;
    namespace
    {
        struct Parameters
        {
            Parameters(int channel_in, int height_in, int width_in, int channel_out,
                       int height_filter, int width_filter, int stride, int pad_w,
                       int pad_h, int repr, int NetworkIdx, int LayerIdx)
                : channel_in(channel_in),
                  height_in(height_in),
                  width_in(width_in),
                  channel_out(channel_out),
                  stride(stride),
                  height_filter(height_filter),
                  width_filter(width_filter),
                  pad_w(pad_w),
                  pad_h(pad_h),
                  repr(repr),
                  NetworkIdx(NetworkIdx),
                  LayerIdx(LayerIdx){};
            int channel_in, channel_out;
            int width_in, height_in;
            int width_filter, height_filter;
            int stride;
            int pad_w, pad_h;
            int repr;
            int NetworkIdx;
            int LayerIdx;

            friend std::ostream &operator<<(std::ostream &os, const Parameters &params)
            {
                return os << std::endl
                          << "| "
                          << "channel_in  :" << params.channel_in << ","
                          << "w_in : " << params.width_in << ","
                          << "h_in : " << params.height_in << "|" << std::endl
                          << "| "
                          << "channel_out :" << params.channel_out << ","
                          << "w_ft : " << params.width_filter << "  ,"
                          << "h_ft : " << params.height_filter << "|" << std::endl
                          << "| "
                          << "stride      :" << params.stride << ","
                          << "pad_w : " << params.pad_w << ","
                          << "pad_h : " << params.pad_h << "|" << std::endl;
            }
        };

        class ConvTest : public ::testing::TestWithParam<Parameters>
        {
        public:
            int channel_in, channel_out;
            int width_in, height_in;
            int width_filter, height_filter;
            int width_out, height_out;
            int stride;
            int pad_w, pad_h;
            int repr;
            int NetworkIdx;
            int LayerIdx;
            vector<Matrix> dmat;
            Matrix data_in;

            virtual void SetUp()
            {
                std::cout << GetParam() << std::endl;
                channel_in = GetParam().channel_in;
                width_in = GetParam().width_in;
                height_in = GetParam().height_in;
                channel_out = GetParam().channel_out;
                width_filter = GetParam().width_filter;
                height_filter = GetParam().height_filter;
                stride = GetParam().stride;
                pad_w = GetParam().pad_w;
                pad_h = GetParam().pad_h;
                repr = GetParam().repr;
                NetworkIdx = GetParam().NetworkIdx;
                LayerIdx = GetParam().LayerIdx;
                data_in = Matrix::Random(width_in * height_in * channel_in, 1);
                width_out = (width_in - width_filter + 2 * pad_w) / stride + 1;
                height_out = (height_in - height_filter + 2 * pad_h) / stride + 1;
            }
            virtual void TearDown()
            {
                // Need to exist while SetUp function used
            }
        };

        auto ParamToString = [](const auto &param_info)
        {
            string name = "c_w_h_IN_" + to_string(param_info.param.channel_in) + "_" +
                          to_string(param_info.param.width_in) + "_" +
                          to_string(param_info.param.height_in) + "_" + "_Filter_" +
                          to_string(param_info.param.channel_out) + "_" +
                          to_string(param_info.param.width_filter) + "_" +
                          to_string(param_info.param.height_filter) + "_";
            if (param_info.param.stride != 1)
                name = name + "_stride_" + to_string(param_info.param.stride);
            if (param_info.param.pad_w != 0)
                name = name + "_padw_" + to_string(param_info.param.pad_w);
            if (param_info.param.pad_h != 0)
                name = name + "_padh_" + to_string(param_info.param.pad_h);
            if (param_info.param.repr != 1)
                name = name + "_flatten";
            if (param_info.param.repr == 3)
                name = name + "_Lazy";

            return name;
        };

        class DummyNetworkConv : public ConvTest
        {
        public:
            virtual void SetUp()
            {
                ConvTest::SetUp();
                // Need to exist while SetUp function used
            }
            virtual void TearDown()
            {
                // Need to exist while SetUp function used
            }
        };

        TEST_P(DummyNetworkConv, ConvBasic)
        {
            Network dnn;
            std::shared_ptr<Layer> conv =
                std::make_shared<Conv>(channel_in, height_in, width_in, channel_out,
                                       height_filter, width_filter, stride, pad_w, pad_h);
            dnn.add_layer(conv);
            std::vector<std::vector<long double>> param;
            string NetworkName = "nn" + to_string(NetworkIdx);
            string LayerName = "conv" + to_string(LayerIdx);
            param.push_back(genParameter(channel_in, height_in, width_in, channel_out, height_filter, width_filter));
            dnn.set_parameters(param);
            dnn.forward(data_in);
            auto output = dnn.output();
        }

        INSTANTIATE_TEST_SUITE_P(
            MNISTLayers, DummyNetworkConv,
            ::testing::Values(Parameters(1, 28, 28, 1, 5, 5, 2, 0, 0, 1, 1, 1),
                              Parameters(1, 28, 28, 1, 5, 5, 2, 0, 0, 2, 1, 1)),
            ParamToString);
        INSTANTIATE_TEST_SUITE_P(
            CifarLayers, DummyNetworkConv,
            ::testing::Values(Parameters(3, 32, 32, 3, 3, 3, 1, 1, 1, 2, 2, 1),
                              Parameters(3, 32, 32, 3, 3, 3, 1, 1, 1, 3, 2, 1),
                              Parameters(17, 16, 16, 3, 3, 3, 1, 0, 0, 2, 2, 2),
                              Parameters(17, 16, 16, 3, 3, 3, 1, 0, 0, 3, 2, 2),
                              Parameters(14, 7, 7, 7, 3, 3, 1, 0, 0, 2, 2, 3),
                              Parameters(14, 7, 7, 7, 3, 3, 1, 0, 0, 3, 2, 3)),
            ParamToString);
        class RealNetworkConv : public ConvTest
        {
        public:
            virtual void SetUp()
            {
                ConvTest::SetUp();
                // Need to exist while SetUp function used
            }
            virtual void TearDown()
            {
                // Need to exist while SetUp function used
            }
        };

        class AvgPoolTest : public ::testing::TestWithParam<Parameters>
        {
        public:
            int channel_in, channel_out;
            int width_in, height_in;
            int width_filter, height_filter;
            int width_out, height_out;
            int stride;
            int pad_w, pad_h;
            int repr;
            int NetworkIdx;
            int LayerIdx;
            vector<Matrix> dmat;
            Matrix data_in;

            virtual void SetUp()
            {
                std::cout << GetParam() << std::endl;
                channel_in = GetParam().channel_in;
                width_in = GetParam().width_in;
                height_in = GetParam().height_in;
                channel_out = GetParam().channel_out;
                width_filter = GetParam().width_filter;
                height_filter = GetParam().height_filter;
                stride = GetParam().stride;
                pad_w = GetParam().pad_w;
                pad_h = GetParam().pad_h;
                repr = GetParam().repr;
                NetworkIdx = GetParam().NetworkIdx;
                LayerIdx = GetParam().LayerIdx;
                data_in = Matrix::Random(width_in * height_in * channel_in, 1);
                width_out = ceil(float(width_in - width_filter + 2 * pad_w) / stride) + 1;
                height_out =
                    ceil(float(height_in - height_filter + 2 * pad_h) / stride) + 1;
            }
            virtual void TearDown()
            {
                // Need to exist while SetUp function used
            }
        };

        TEST_P(AvgPoolTest, avePooling1)
        {
            //! function prototype
            //! AvePooling(int channel_in, int height_in, int width_in,
            //           int height_pool, int width_pool, int stride = 1)
            Network dnn;
            std::shared_ptr<Layer> avePooling = std::make_shared<AvePooling>(
                channel_in, height_in, width_in, height_filter, width_filter,
                stride); // usally 2x2 avgpooling and stride=2 maybe

            dnn.add_layer(avePooling);
            dnn.forward(data_in);
        }

        INSTANTIATE_TEST_SUITE_P(
            MNISTLayers, AvgPoolTest,
            ::testing::Values(Parameters(1, 28, 28, 1, 2, 2, 2, 0, 0, 2, 1, 1)),
            ParamToString);

        INSTANTIATE_TEST_SUITE_P(
            CifarLayers, AvgPoolTest,
            ::testing::Values(Parameters(128, 32, 32, 1, 2, 2, 2, 0, 0, 2, 1, 1),
                              Parameters(83, 14, 14, 1, 2, 2, 2, 0, 0, 2, 1, 1),
                              Parameters(163, 5, 5, 1, 2, 2, 2, 0, 0, 2, 1, 1)),
            ParamToString);
    } // namespace
} // namespace cppnn
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
            Parameters(int height_in, int width_in, int NetworkIdx, int LayerIdx)
                : height_in(height_in),
                  width_in(width_in),
                  NetworkIdx(NetworkIdx),
                  LayerIdx(LayerIdx){};
            int width_in, height_in;
            int NetworkIdx, LayerIdx;

            friend std::ostream &operator<<(std::ostream &os, const Parameters &params)
            {
                return os << std::endl
                          << "| "
                          << "w_in : " << params.width_in << ","
                          << " h_in : " << params.height_in << "|" << std::endl;
            }
        };

        class FCTest : public ::testing::TestWithParam<Parameters>
        {
        public:
            int width_in, height_in;
            int NetworkIdx, LayerIdx;
            Matrix dmat;
            Matrix data_in;

            virtual void SetUp()
            {
                std::cout << GetParam() << std::endl;
                width_in = GetParam().width_in;
                height_in = GetParam().height_in;
                NetworkIdx = GetParam().NetworkIdx;
                LayerIdx = GetParam().LayerIdx;
                data_in = Matrix::Random(width_in, 1);
            }
            virtual void TearDown()
            {
                // Need to exist while SetUp function used
            }
        };

        auto ParamToString = [](const auto &param_info)
        {
            string name = "w_h_IN_" + to_string(param_info.param.width_in) + "_" +
                          to_string(param_info.param.height_in) + "_" + "_Filter";
            return name;
        };

        TEST_P(FCTest, FCBasic)
        {
            Matrix fc_weight = Matrix::Random(width_in, height_in);
            const auto output = fc_weight.transpose() * data_in;
        }

        INSTANTIATE_TEST_SUITE_P(NN1, FCTest,
                                 ::testing::Values(Parameters(10, 163 * 3 * 3, 2, 1),
                                                   Parameters(40, 144, 1, 1),
                                                   Parameters(10, 40, 1, 2)),
                                 ParamToString);

    } // end of namespace
} // namespace cppnn
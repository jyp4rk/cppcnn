# Copyright 2022 Jaiyoung Park
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.18)
include(${PROJECT_SOURCE_DIR}/cmake/CPM.cmake)
include(FetchContent)

cpmaddpackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG release-1.11.0
  OPTIONS
      "INSTALL_GTEST OFF"
      "BUILD_GMOCK ON"
)
ADD_EXECUTABLE(ConvTest ConvTest.cpp)
ADD_EXECUTABLE(FCTest FCTest.cpp)
ADD_EXECUTABLE(InferenceTest InferenceTest.cpp)

enable_testing()
include_directories(${googletest_SOURCE_DIR}/include ${googletest_SOURCE_DIR}
                    ${googletest_SOURCE_DIR}/googlemock/include include)

target_link_libraries(FCTest cppnn gtest gtest_main)
target_link_libraries(ConvTest cppnn gtest gtest_main)
target_link_libraries(InferenceTest cppnn gtest gtest_main)

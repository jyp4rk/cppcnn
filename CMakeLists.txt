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
project(cppnn LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB nn_srcs 
			./nn/*.cpp 
			./nn/Layer/*.cpp
		)

find_package(ZLIB REQUIRED)
add_subdirectory(test)
include(cmake/CPM.cmake)
include(cmake/eigen.cmake)
add_library(cppnn ${nn_srcs})

target_link_libraries(cppnn PUBLIC cnpy Eigen)
target_include_directories(cppnn
  PUBLIC nn nn/Layer
)

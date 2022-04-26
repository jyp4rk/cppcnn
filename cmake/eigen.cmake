cmake_minimum_required(VERSION 3.18)
include(FetchContent)

FetchContent_Declare(
    cnpy
    GIT_REPOSITORY https://github.com/rogersce/cnpy
)
cpmaddpackage(
  NAME Eigen
  URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"
  DOWNLOAD_ONLY YES
  )

if(Eigen_ADDED)
  add_library(Eigen INTERFACE)
  target_include_directories(Eigen INTERFACE ${Eigen_SOURCE_DIR})
endif()

FetchContent_MakeAvailable(cnpy)

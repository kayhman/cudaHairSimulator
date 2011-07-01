cmake_minimum_required(VERSION 2.8)

project(smokeRendering)

find_package(Cuda)
find_package(OpenGL)

add_subdirectory(src)
#!/bin/sh

cd shaders
glslangValidator -V shader.vert
glslangValidator -V shader.frag
cd ..
g++ -O3 --std=c++17 -lvulkan -lglfw model.cpp

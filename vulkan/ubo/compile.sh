#!/bin/sh

cd shaders
glslangValidator -V shader.vert
glslangValidator -V shader.frag
cd ..
g++ -O0 -g3 --std=c++17 -lvulkan -lglfw ubo.cpp

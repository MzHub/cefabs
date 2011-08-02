#!/bin/sh
cmake -E make_directory build
cd build
cmake ..
make
cpack

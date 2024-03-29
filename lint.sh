#!/bin/bash

# It seems like "IndentPragma" in clang-format was abandoned.
# Following this block post:
# https://medicineyeh.wordpress.com/2017/07/13/clang-format-with-pragma/

set -eo pipefail

# get git root dir
gitroot=$(git rev-parse --show-toplevel)

# run python formatter in the git root directory
(cd $gitroot && black .)

# get all c/c++/cuda files in the git src directory
srcdir = $gitroot/src
files=$(find $srcdir -type f -name "*.c" -o -name "*.cpp" -o -name "*.cu" -o -name "*.h" -o -name "*.hpp" -o -name "*.cuh")

for file in $files; do
    echo "Linting $file"
    # Replace "#pragma unroll" by "//#pragma unroll"
    sed -i 's/#pragma unroll/\/\/#pragma unroll/g' $file
    # Do format
    clang-format -i $file
    # Replace "// *#pragma unroll" by "#pragma unroll"
    sed -i 's/\/\/ *#pragma unroll/#pragma unroll/g' $file
done

#!/bin/bash

# It seems like "IndentPragma" in clang-format was abandoned.
# Following this block post:
# https://medicineyeh.wordpress.com/2017/07/13/clang-format-with-pragma/

set -eo pipefail

# get git root dir
GITROOT=$(git rev-parse --show-toplevel)

# run python formatter in the git root directory
(cd $GITROOT && black .)

# get all c/c++/cuda files in the git src directory
SRCDIR=$GITROOT/src
FILES=$(find $SRCDIR -type f -name "*.c" -o -name "*.cpp" -o -name "*.cu" -o -name "*.h" -o -name "*.hpp" -o -name "*.cuh")

for FILE in $FILES; do
    echo "Linting $FILE"
    # Replace "#pragma unroll" by "//#pragma unroll"
    sed -i 's/#pragma unroll/\/\/#pragma unroll/g' $FILE
    # Do format
    clang-format -i $FILE
    # Replace "// *#pragma unroll" by "#pragma unroll"
    sed -i 's/\/\/ *#pragma unroll/#pragma unroll/g' $FILE
done

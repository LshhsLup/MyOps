#!/bin/bash
# Format C++/CUDA source files with clang-format

set -e

echo "Formatting C++/CUDA files..."

for f in $(find csrc -name "*.cu" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" 2>/dev/null); do
    echo "  Formatting $f"
    clang-format -i "$f"
done

echo "Done!"
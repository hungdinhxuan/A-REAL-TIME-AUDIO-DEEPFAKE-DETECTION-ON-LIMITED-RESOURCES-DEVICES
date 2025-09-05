#!/bin/bash

echo "Testing file reading..."

# Test 1: Check if file exists and is readable
if [[ -f "data/replay_cl_250610/small_set.txt" ]]; then
    echo "File exists"
else
    echo "File does not exist"
    exit 1
fi

# Test 2: Check file size
file_size=$(wc -c < "data/replay_cl_250610/small_set.txt")
echo "File size: $file_size bytes"

# Test 3: Check number of lines
line_count=$(wc -l < "data/replay_cl_250610/small_set.txt")
echo "Line count: $line_count"

# Test 4: Try to read first few lines
echo "First 3 lines:"
head -3 "data/replay_cl_250610/small_set.txt"

# Test 5: Try to extract first column
echo "First 3 first columns:"
awk '{print $1}' "data/replay_cl_250610/small_set.txt" | head -3

# Test 6: Check if awk is working
echo "Testing awk with a simple file:"
echo "test line 1" > test_file.txt
echo "test line 2" >> test_file.txt
awk '{print $1}' test_file.txt
rm test_file.txt

echo "Debug test completed" 
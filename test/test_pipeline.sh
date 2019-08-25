#!/bin/bash
# Integration test for pipeline

# get the directory of this file
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR || exit
touch integration_test_result.txt
mkdir -p ./test_output/archive
rm -r ./test_output/archive/*
python ../cftm/pipeline.py 1 1 1 -p '../test/test_path.yaml' -o 30000 -l 300 -r 1 102 10 -s 2 -c 300 -i 900

# clean up
if [ "$(ls -1 ../test/test_output/archive/20*/* | wc -l)" -ne 6 ]; then
   echo "archive failed!" > ./integration_test_result.txt
   rm -r ./test_output/archive/*
else
   echo "archive succeed!" > ./integration_test_result.txt
   rm -r ./test_output/archive/*
fi
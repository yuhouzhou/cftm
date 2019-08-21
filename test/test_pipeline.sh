#!/bin/sh
# Integration test for pipeline

# get the directory of this file
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR
touch integration_test_result.txt
rm -r ./test_output/archive/*
python ../cftm/pipeline.py 1 1 1 -p '../test/test_path.yaml' -o 30000 -a 300 -r 25 35 -s 2

# clean up
if [ "$(ls -1 ../test/test_output/archive/20*/* | wc -l)" -ne 5 ]; then
   echo "archive failed!" > ./integration_test_result.txt
   rm -r ./test_output/archive/*
else
   echo "archive succeed!" > ./integration_test_result.txt
   rm -r ./test_output/archive/*
fi
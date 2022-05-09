#!/bin/bash

cuda_file_name="nd4j-cuda-11.2-1.0.0-M1.1-linux-x86_64.jar"

echo "downloading $cuda_file_name"

wget https://repo1.maven.org/maven2/org/nd4j/nd4j-cuda-11.2/1.0.0-M1.1/${cuda_file_name}

echo "unzipping $cuda_file_name"

unzip ${cuda_file_name}


dest_dir="/usr/local/cuda/lib64/"
search_dir=./org/nd4j/nativeblas/linux-x86_64/
for entry in "$search_dir"/*
do
  mv "$entry" dest_dir
  echo "moving $entry to $dest_dir"
done
#!/bin/bash

port=$1
if [ -z "$port" ]; then
  echo 'specify your port number'
  echo "ex) $0 10100"
  exit 1
fi

docker run -it --rm \
  --name sayalab-$(id -u $USER) \
  --gpus 1 \
  -p ${port}:8888 \
  -u $(id -u $USER):1005 \
  -v /etc/group:/etc/group:ro \
  -v /etc/passwd:/etc/passwd:ro \
  -v "$(pwd)":/work \
  -v /raid/projects/ko-sayalab/arima/data:/work/data \
  sayalab:cuda114

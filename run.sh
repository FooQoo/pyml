#!/bin/sh

IMAGE_NAME=$1
shift

if [ "${IMAGE_NAME}" = "" ]; then
  exit 1
fi

docker run --rm -it \
  -v $(pwd):/usr/local/${IMAGE_NAME} \
  -w /usr/local/${IMAGE_NAME}/src/ \
  ${IMAGE_NAME} "$@"
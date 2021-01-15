#!/bin/bash

# get local directory
DIR=$(readlink -f $(dirname ${BASH_SOURCE[0]}))

export PATH="${DIR}/auto/bin:${PATH}"
export PYTHONPATH="${DIR}:${PYTHONPATH}"
export LANG=en_US.utf8

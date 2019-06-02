#!/bin/bash

FWDIR="$(cd `dirname $0`; pwd)"
WORK_DIR="$FWDIR/.."

export PYTHONPATH="$WORK_DIR/python/:$PYTHONPATH"

executable=$1

python $executable "${@:2}"

#!/bin/bash


# set following parameters according to the actual specs of the test machine
numactl -C 0-47 -m 0 python $(dirname "$0")/run.py
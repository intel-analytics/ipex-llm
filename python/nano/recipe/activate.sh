#!/bin/bash -e -x

echo "=====activate===="
is_tf=$(conda list bigdl-nano.tf | grep bigdl-nano.tf) || is_tf="no"

if [ "${is_tf}" = "no" ];then
    echo "Use bigdl-nano-run support for pytorch"
    bigdl-nano-run
else
    echo "find bigdl-nano-run support for tensorflow"
    bigdl-nano-run -t
fi

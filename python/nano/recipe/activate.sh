#!/bin/bash -e -x

echo "=====activate===="
is_tf=$(conda list bigdl-nano.tf | grep bigdl-nano.tf) || is_tf="no"

if [ "${is_tf}" = "no" ];then
    echo "Use bigdl-nano-init support for pytorch"
    bigdl-nano-init
else
    echo "find bigdl-nano-init support for tensorflow"
    bigdl-nano-init -t
fi

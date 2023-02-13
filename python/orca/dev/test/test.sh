#!/usr/bin/env bash

function stop_ray() {
    if [ $1 = "ray" ]; then
        echo "Trying to stop any activate ray context..."
        ray stop -f
    else 
        echo "Backend is not ray, skipping"
    fi
}

# backend passed as the first argument, either "ray" or "spark"
# if no argument is provided, default to be "spark"
argc=$#
if [ $argc -eq 0 ]; then
    backend="spark"
else
    backend=$1
fi
echo "Start Orca NCF tutorial Test - $backend backend"

stop_ray $backend
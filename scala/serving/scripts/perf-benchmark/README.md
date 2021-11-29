# Cluster Serving Performance Benchmarking #
This is a simple script that benchmarks the end-to-end cluster serving performance in terms of throughput (images per second).

The script calculates the end-to-end throughput by repeatedly pushing a test image to cluster serving and measures the time from starting request to receive inference results.

1. Before running the script, please install the required python packages by `pip install -r requirement.yml`

2. Run benchmark with `python e2e_throughput.py -c /path/to/cluster/serving/config.yaml -i path/to/test/image` (If TLS is used, please also pass key and cert directory by `-k path/to/key/directory`)

	By default, the image is pushed 10000 times with 10 multiple processes. You can also adjust the push number and multiprocess number by adding argument `-n` and `-p` while running the command. 
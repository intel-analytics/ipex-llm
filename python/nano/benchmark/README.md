# How to add bigdl-nano performance tests
This doc is to help you add bigdl-nano performance test.
## Source Code
You could follow the `pytorch-cat-vs-dog.py` to add other benchmark test. For example, to test the throughput, time the training process, and calculate the throughput (img/s) by `throughput = epochs * train_size / timing`

If you hope to save the test result, add the module to save result (e.g. save to `csv` file at `BigDL/python/nano/benchmark/results/` directory by default).

## Script Setting
Based on test program, set up the path environment in `Enviroment Settings` part , and replace the boot-up command at `Boot-up commands` part of `start_benchmark.sh` at `BigDL/python/nano/dev`.

If you want to test several benchmark tests once, add all boot up commands to `Boot-up commands` part at `start_benchmark.sh`

<font size=5>

# **Benchmark Tool**


This topic demonstrates how to use the benchmark tool, which helps developers and users to benchmark the training and inferencing performance on their own platform. 

## Run the tool

To run the tool, you can directly use the command line:
```python
python benchmark-chronos.py
```

## Capture hardware information

> CPU_info()

    Display some hardware information. These hardware configurations are related to the performance on the platform.
    Below is sample output:

```
>>>>>>>>>>>>>>>>>>>> Hardware Information >>>>>>>>>>>>>>>>>>>>
CPU architecture: x86_64
CPU model name: Intel(R) Core(TM) i7-3960X CPU @ 3.30GHz
Logical Core(s): 12
Physical Core(s): 6
Physical Core(s) per socket: 6
Socket(s): 1
CPU usage: 14.7%
CPU MHz: 2052.23
CPU max MHz: 3900.00
CPU min MHz: 1200.00
Total memory: 31.34GB
Available memory: 21.43GB
Support avx512f : ✘
Support avx512_bf16 : ✘
Support avx512_vnni : ✘
<<<<<<<<<<<<<<<<<<<< Hardware Information <<<<<<<<<<<<<<<<<<<<  
```

## Diagnose environment variables
> check_nano()

    Detect whether nano environment variables are set properly and supply necessary suggestions.
    Below are sample output for proper and improper environment settings:

```
· Proper environment settings:

>>>>>>>>>>>>>>>>>>>> Environment Variables >>>>>>>>>>>>>>>>>>>>
LD_PRELOAD  enabled ✔
tcmalloc  enabled ✔
Intel OpenMp  enabled ✔
TF  enabled ✔
<<<<<<<<<<<<<<<<<<<< Environment Variables <<<<<<<<<<<<<<<<<<<<
```

```
· Improper environment settings:

>>>>>>>>>>>>>>>>>>>> Environment Variables >>>>>>>>>>>>>>>>>>>>
LD_PRELOAD  enabled ✔
tcmalloc  enabled ✔
Intel OpenMp : OMP_NUM_THREADS  not enabled ✘
TF  enabled ✔
 
++++++++++++++++++++ Suggested change:  ++++++++++++++++++++
export OMP_NUM_THREADS=6
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 

<<<<<<<<<<<<<<<<<<<< Environment Variables <<<<<<<<<<<<<<<<<<<<
```


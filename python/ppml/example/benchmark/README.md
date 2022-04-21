# PPML Benchmark Test

## Python API Benchmark Test

### Set up
#### Install Python packages
* Supported Python version: <= 3.6

To set up the benchmark test, first install `pyspark` corresponded to the BigDL spark version to test.

e.g. If the BigDL jar has name `bigdl-ppml-spark_2.4.6`, use `pip install pyspark==2.4.6` to install pyspark.

The BigDL spark version could be set in `pom.xml` before jar is packaged.

Other packages, e.g. numpy, pandas, has no strict version requirement so far, just install the missing packages manually.
#### Prepare PPML Jar
Normally, PPML jar could be setup by 
```bash
cd BigDL/scala/ppml &&
mvn clean package -DskipTests &&
mv target/bigdl-ppml-*-jar-with-dependencies.jar ../../dist/lib/ # this copy the jar to BigDL/dist/lib directory
```
The `BigDL/dist/lib` directory is the path which python code would search for its Jar dependency.

If package fails due to some dependencies of PPML, try `cd BigDL/scala && ./make-dist.sh` to install the dependencies to local. This usually happens during the first setup and the dependencies packages are not available. 

#### Set Python Environment
If PPML is installed by pip, this step could be skipped.

If PPML is not installed, benchmark tests could still move forward by setting `PYTHONPATH`.
```
export PYTHONPATH=$PYTHONPATH:/path/to/BigDL/python/dllib/src:/path/to/BigDL/python/ppml/src
```
If the source code is available, a prepared script could also set above `PYTHONPATH` automatically
```
cd BigDL/python/ppml/dev &&
source prepare_env.sh
```
### Testing
For testing steps, please refer to the specific directory of the test to carry out.
## Scala API Benchmark Test
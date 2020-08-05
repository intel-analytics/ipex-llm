# 天池大赛 Cluster Serving Quick Start 中文版

## 安装
前置需求：Python，Redis 5.0.5，Flink 1.11.0，在官网安装即可，安装后设置环境变量`REDIS_HOME=/path/to/redis-5.0.5, FLINK_HOME=/path/to/flink-1.11.0`
### Python 依赖
可复制以下内容到`requirement.txt`并使用`pip install -r requirements.txt`安装，也可在遇到找不到模块的报错时安装相应依赖
```
redis
pyyaml
httpx
pyarrow
pyspark
```
### 安装Redis
```
$ wget http://download.redis.io/releases/redis-5.0.5.tar.gz
$ tar xzf redis-5.0.5.tar.gz
$ cd redis-5.0.5
$ make
```
### 安装Flink
```
$ wget https://archive.apache.org/dist/flink/flink-1.11.0/flink-1.11.0-bin-scala_2.11.tgz
$ tar xzf flink-1.11.0-bin-scala_2.11.tgz
```
### 安装Cluster Serving
下载`analytics-zoo-xxx-cluster-serving-all.zip`[下载地址](https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/zoo/analytics-zoo-bigdl_0.10.0-spark_2.4.3/0.9.0-SNAPSHOT/analytics-zoo-bigdl_0.10.0-spark_2.4.3-0.9.0-20200804.085942-62-cluster-serving-all.zip)，解压后进入`cluster-serving`目录，运行`source cluster-serving-prepare.sh`

若要使用同步API运行`java -jar analytics-zoo-xxx-http.jar`启动同步服务

如果要在IDE里面运行，需要在IDE环境变量中设置`PYTHONPATH=/path/to/analytics-zoo-xxx-python-api.zip`

若想以当前目录为工作目录，则一切就绪

若想以其他目录为工作目录，进入目录后，设置环境变量`export CS_PATH=/path/to/analytics-zoo-xxx-cluster-serving-all/cluster-serving`（解压zip包的目录），之后命令行运行`cluster-serving-init`
## 配置
修改`config.yaml`，配置模型路径为包含模型的文件夹路径，样例如下，假设用户模型为Tensorflow SavedModel模型，结构为
```
my-pro 
  | -- my-model
    | -- saved_model.pb
    | -- variables
      | -- xxx
```
则`config.yaml`内容应为
```
model:
  path: path/to/my-pro/my-model
```
## 启动
在工作目录，命令行运行`cluster-serving-start`

## 推理
使用Python API，传入符合模型格式的输入，**并注意末尾加上小数点，代表float格式**，样例如下，假设Redis启动host为"localhost"，port为"6379"，[同步服务](#安装)启动url为"127.0.0.1:10020"，模型输入为一维，有两个常数，则推理脚本代码如下
```
    input_api = InputQueue(host="localhost", port="6379", sync=True, frontend_url="http://127.0.0.1:10020")
    s = '''{
          "instances": [
            {
              "t": [1.0, 2.0]
            }
          ]
        }'''
    a = input_api.predict(s)
    print(a)
```

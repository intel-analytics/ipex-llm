# 天池大赛 Cluster Serving Quick Start 中文版

## 安装
前置需求：Linux-x86_64，Python，Redis 5.0.5，Flink 1.11.0，在官网安装即可，安装后设置环境变量`REDIS_HOME=/path/to/redis-5.0.5, FLINK_HOME=/path/to/flink-1.11.0`

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
下载天池提供的`whl`安装包，运行`pip install analytics-zoo-*.whl`进行安装，安装后运行`cluster-serving-init`，可以看到生成的`zoo.jar`依赖包，`analytics-zoo-xxx-http.jar`同步服务包，`config.yaml`配置文件

#### 对于下载安装包缓慢的选手
对于在`cluster-serving-init`中下载缓慢的问题，天池提供了`analytics-zoo-*-serving.jar`和`analytics-zoo-*-http.jar`在内网，选手可以下载到工作目录，**并将`analytics-zoo-*-serving.jar`重命名为`zoo.jar`**，命令`mv analytics-zoo-*-serving.jar zoo.jar`，再运行`cluster-serving-init`可以看到生成的`config.yaml`配置文件

当可以看到`zoo.jar, analytics-zoo-*-http.jar, config.yaml`时，即为安装完成

运行`java -jar analytics-zoo-xxx-http.jar`启动同步服务

PS: `zoo.jar`为集成了Flink分布式推理代码的依赖，`cluster-serving-start`会根据工作目录的配置文件`config.yaml`启动`zoo.jar`中的代码，`analytics-zoo-*-http.jar`为同步服务的依赖，本次天池大赛中不需要用到PS内的信息，对Cluster Serving的更多了解请移步[github官方文档](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/ClusterServingGuide/ProgrammingGuide.md)
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
使用同步API，需要传入符合模型格式的输入，**并且注意数据类型为float，即末尾加上小数点代表float格式**，样例如下，

假设Redis启动host为"localhost"，port为"6379"，[同步服务](#安装)启动url为"127.0.0.1:10020"，模型输入为一维，有两个常数，则推理脚本代码如下
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

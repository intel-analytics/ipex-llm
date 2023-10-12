## Use LLM on docker

### Install docker

New users can quickly get started with Docker using this [official link](https://www.docker.com/get-started/).

For Windows users, make sure Hyper-V is enabled on your computer. The instructions for installing on Windows can be accessed from [here](https://docs.docker.com/desktop/install/windows-install/).

### Pull and run bigdl-llm-cpu image

To pull image from hub, you can execute command on console:
```bash
docker pull intelanalytics/bigdl-llm-cpu:2.4.0:SNAPSHOT
```

To run the image and do inference, you could refer to this [documentation](https://github.com/intel-analytics/BigDL/tree/main/docker/llm/inference/cpu/docker#use-the-image-for-doing-cpu-inference). You could also learn how to use built-in chat.py to start a conversation with a specified model from [here](https://github.com/intel-analytics/BigDL/tree/main/docker/llm/inference/cpu/docker#use-chatpy).

### Use the image for development

You could start a jupyter-lab serving to explore bigdl-llm-tutorial which can help you build a more sophisticated Chatbo.

To start serving,  run the script under '/llm':
```bash
cd /llm
./start-notebook.sh [--port EXPECTED_PORT]
```
You could assign a port to serving, or the default port 12345 will be assigned.

If you use host network mode when booted the container, after successfully running service, you can access http://127.0.0.1:12345/lab to get into tutorial, or you should bind the correct ports between container and host. 
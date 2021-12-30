# BigDL orca Developer Container

This docker image contain the basic development environment for BigDL-Orca. You can build a new container with the Dockerfile or simply start from a pre-build image. 

## Quick Start

First, build the image with the Dockerfile

```shell
cd BigDL/python/orca
docker build -t orca-dev:latest -f ./orca-dev.dockerfile
```

You can run a new container with the image:

```shell
docker run --name orca-dev -it orca-dev:latest
```

with the args `-it`, you will attach to a pseudo-TTY in the container. 

You can also run a rebuild container image from `vankyle2go/orca-dev`, which is build from the `Dockerfile` with default build args. 

```shell
docker run --name orca-dev -it vankyle2go/orca-dev:latest
```

## OpenSSH Server Support

The OpenSSH Server is also installed in the container, you can use `-p`  when running  `docker run` to expose the 22 port of container to the host and access the container with SSH client. For example:

```shell
docker run --name orca-dev -it \
-p 8022:22 \
orca-dev:latest
```

then you can access the 8022 port of the host IP address with SSH clients to connect to the container after you run `service ssh start` in the container. 

## Reduce Disk Usage in Root Partition

As most of our servers usually suffer from lacks of disk space of root partition, I personally recommend using  `-v` to mount a directory in other partition into the container when running the `docker run` command. For example:

```shell
docker run --name orca-dev -it \
-p 8022:22 \
-v /path/on/other/partition:/data \
orca-dev:latest
```

However, it may cause problems if you directly mount some empty folder to the home directory of user in container(by default, `/root/`) , where the maven repository storage in. To mount the directory which was configured during the build of image, you can copy these files from a temporary container to the host, and then mount them to the container you are using for development, so that the container can rely on these files to function properly. Here are some reference steps:

First, define the directory on the host that you want to storage files, in this example , we use `/disk1` : 

```shell
root_fs_dir=/disk1
image=orca-dev:latest
container_name=orca-dev
```

Then run up a temporary container to copy the files. 

```shell
docker run -d --name temp-fs $image tail -f /dev/null
docker cp temp-fs:/root $root_fs_dir/root
```

Finally, mount the copied directory to the development container:

```shell
docker run --name $container_name -it \
-p 8022:22 \
-v $root_fs_dir/root:/root
$image
```

Now you have a container with the `/root` mounted to a folder on host. 

## Build Behind Proxy

If you are behind a proxy, you might need to set `HTTP_PROXY` and `HTTPS_PROXY` variable when running `docker build`.  

```shell
docker build --build-arg HTTP_PROXY=<protocol>://<host>:<port> --build-arg HTTPS_PROXY=<protocol>://<host>:<port> -t orca-dev:latest -f orca-dev.dockerfile .
```

These settings will be add to the `/etc/profile.d/02-proxy.sh` and will keep effect when you use. 

However, these settings will **NOT** change the maven configuration, which is defined at `/etc/maven/settings.xml` . You may need to configure the proxy for maven manually after build. Or you can simply modify the `Dockerfile` to add a `COPY` or `ADD` command to provide a new `settings.xml` to the container.

Here are some steps you can refer to change the maven settings. 

First, you can get a template from a temporary container:

```shell
image=orca-dev:latest
docker run -d --name temp-fs $image tail -f /dev/null
docker cp temp-fs:/etc/maven/setttings.xml ./settings.xml
```

then you can modify the `settings.xml ` as you like, especially the proxy settings. 

```xml
<proxies>
    <proxy>
      <id>optional</id>
      <active>true</active>
      <protocol>http</protocol>
      <username>proxyuser</username>
      <password>proxypass</password>
      <host>proxy.host.net</host>
      <port>80</port>
      <nonProxyHosts>local.net|some.host.com</nonProxyHosts>
    </proxy>
</proxies>
```

Next, modify the `Dockerfile`, add a `COPY` command to add your settings.xml to the right place. Or you can use the following command to add the `settings.xml` after you run up a container.

```shell
docker cp settings.xml <your-container-name>:/etc/mavem/setttings.xml
```



## How it Works

This `Dockerfile` refers to [dockerfile](https://github.com/jupyter/docker-stacks/tree/master/base-notebook) of  [jupyter/base-notebook](https://hub.docker.com/r/jupyter/base-notebook) to build an environment with conda. Other components including scala-2.12.0 are also included. Network proxy and environment variables like `JAVA_HOME` is also configured  automatically. 


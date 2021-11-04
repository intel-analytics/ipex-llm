#!/bin/bash
set -x


BLUE='\033[1;34m'
NC='\033[0m'
occlum_glibc=/opt/occlum/glibc/lib/

init_occlum_instance() {
    # Init Occlum instance
    rm -rf occlum_spark && mkdir occlum_spark
    cd occlum_spark
    occlum init
    new_json="$(jq '.resource_limits.user_space_size = "16000MB" |
        .resource_limits.max_num_of_threads = 256 |
        .process.default_heap_size = "128MB" |
        .resource_limits.kernel_space_heap_size="256MB" |
        .process.default_mmap_size = "15000MB" |
        .entry_points = [ "/usr/lib/jvm/java-8-openjdk-amd64/bin" ] |
	.env.default = [ "LD_LIBRARY_PATH=/usr/lib/jvm/java-8-openjdk-amd64/lib/server:/usr/lib/jvm/java-8-openjdk-amd64/lib:/usr/lib/jvm/java-8-openjdk-amd64/../lib:/lib","SPARK_CONF_DIR=/opt/spark/conf","SPARK_ENV_LOADED=1","PYTHONHASHSEED=0","SPARK_HOME=/opt/spark","SPARK_SCALA_VERSION=2.12","SPARK_JARS_DIR=/opt/spark/jars","LAUNCH_CLASSPATH=/opt/spark/jars/*","SPARK_CLASSPATH=/opt/spark/jars/*"] |
		.env.untrusted = [ "KUBERNETES_SERVICE_PORT_HTTPS", "KUBERNETES_SERVICE_PORT", "SPARK_EXECUTOR_ID", "HOSTNAME", "SPARK_JAVA_OPT_0", "SPARK_JAVA_OPT_1", "SPARK_JAVA_OPT_2", "KUBERNETES_PORT_443_TCP", "SPARK_APPLICATION_ID", "SPARK_EXECUTOR_CORES", "SPARK_USER", "SPARK_LOCAL_DIRS", "KUBERNETES_PORT_443_TCP_PROTO", "KUBERNETES_PORT_443_TCP_ADDR", "SPARK_EXECUTOR_MEMORY", "KUBERNETES_SERVICE_HOST", "KUBERNETES_PORT", "KUBERNETES_PORT_443_TCP_PORT", "SPARK_DRIVER_URL", "SPARK_EXECUTOR_POD_IP"]' Occlum.json)" && \
    echo "${new_json}" > Occlum.json

}

build_spark() {
    # Copy JVM and class file into Occlum instance and build
    mkdir -p image/usr/lib/jvm
    mkdir -p image/opt/spark
    cp -r /usr/lib/jvm/java-8-openjdk-amd64 image/usr/lib/jvm
    cp /lib/x86_64-linux-gnu/libz.so.1 image/$occlum_glibc
    cp /lib/x86_64-linux-gnu/libtinfo.so.5 image/$occlum_glibc
    cp /lib/x86_64-linux-gnu/librt.so.1 image/$occlum_glibc
    cp /lib/x86_64-linux-gnu/libdl.so.2  image/$occlum_glibc
    cp /lib/x86_64-linux-gnu/libresolv.so.2 image/$occlum_glibc
    cp /lib/x86_64-linux-gnu/libnss*.so.2 image/$occlum_glibc
    cp -rf $SPARK_HOME/* image/opt/spark/
    cp -rf /etc/ssl image/etc/
    cp -rf /proc/cpuinfo image/proc/
    cp -rf /etc/passwd image/etc/
    cp -rf /etc/resolv.conf image/etc/resolv.conf
    cp -rf /etc/hosts image/etc/
    cp -rf /etc/hostname image/etc/
	echo "127.0.0.1 occlum-node" >> image/etc/hosts
    cp -rf /etc/group image/etc/
    occlum build
}

init_occlum_instance
build_spark

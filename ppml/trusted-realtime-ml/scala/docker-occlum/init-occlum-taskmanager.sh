#!/bin/bash
# set -x

occlum_glibc=/opt/occlum/glibc/lib/
HOST_IP=`cat /etc/hosts | grep $HOSTNAME | awk '{print $1}'`
init_instance() {
    # Init Occlum instance
    cd /opt
    # Remove older instance
    rm -rf flink && mkdir flink
    cd flink
    # init occlum
    occlum init
    new_json="$(jq '.resource_limits.user_space_size = "SGX_MEM_SIZE" |
                .resource_limits.kernel_space_heap_size="SGX_KERNEL_HEAP" |
                .resource_limits.max_num_of_threads = "SGX_THREAD"  |
                .process.default_heap_size = "SGX_HEAP" |
                .entry_points = [ "/usr/lib/jvm/java-11-openjdk-amd64/bin" ] |
                .env.default = [ "LD_LIBRARY_PATH=/usr/lib/jvm/java-11-openjdk-amd64/lib/server:/usr/lib/jvm/java-11-openjdk-amd64/lib:/lib:/opt/occlum/glibc/lib/", "OMP_NUM_THREADS=4", "KMP_AFFINITY=verbose,granularity=fine,compact,1,0", "KMP_BLOCKTIME=20" ]' Occlum.json)" && \
    echo "${new_json}" > Occlum.json

    echo "SGX_MEM_SIZE ${SGX_MEM_SIZE}"
    if [[ -z "$SGX_MEM_SIZE" ]]; then
        sed -i "s/SGX_MEM_SIZE/32000MB/g" Occlum.json
    else
        sed -i "s/SGX_MEM_SIZE/${SGX_MEM_SIZE}/g" Occlum.json
    fi

    if [[ -z "$SGX_THREAD" ]]; then
        sed -i "s/\"SGX_THREAD\"/256/g" Occlum.json
    else
        sed -i "s/\"SGX_THREAD\"/${SGX_THREAD}/g" Occlum.json
    fi

    if [[ -z "$SGX_HEAP" ]]; then
        sed -i "s/SGX_HEAP/128MB/g" Occlum.json
    else
        sed -i "s/SGX_HEAP/${SGX_HEAP}/g" Occlum.json
    fi

    if [[ -z "$SGX_KERNEL_HEAP" ]]; then
        sed -i "s/SGX_KERNEL_HEAP/512MB/g" Occlum.json
    else
        sed -i "s/SGX_KERNEL_HEAP/${SGX_KERNEL_HEAP}/g" Occlum.json
    fi
}

build_flink() {
    # Copy JVM and class file into Occlum instance and build
    mkdir -p image/usr/lib/jvm
    cp -r /usr/lib/jvm/java-11-openjdk-amd64 image/usr/lib/jvm
    cp /lib/x86_64-linux-gnu/libz.so.1 image/lib
    cp $occlum_glibc/libdl.so.2 image/$occlum_glibc
    cp $occlum_glibc/librt.so.1 image/$occlum_glibc
    cp $occlum_glibc/libm.so.6 image/$occlum_glibc
    cp $occlum_glibc/libnss_files.so.2 image/$occlum_glibc
    cp -rf /opt/keys image/opt/
    cp -rf /opt/flink-${FLINK_VERSION}/* image/bin/
    cp -rf /opt/flink-${FLINK_VERSION}/conf image/opt/
    cp -rf /etc/java-11-openjdk image/etc/
    cp -rf /etc/hosts image/etc/
    echo "$HOST_IP occlum-node" >> image/etc/hosts
    # cat image/etc/hosts
    cp -rf /etc/hostname image/etc/
    # build occlum
    occlum build
}

#Build the flink occlum instance
init_instance
build_flink

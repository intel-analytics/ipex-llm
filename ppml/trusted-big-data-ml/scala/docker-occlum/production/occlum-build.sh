# default
export container_name=2.5.0-SNAPSHOT-build-container
export image_name=intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum-production:2.5.0-SNAPSHOT
export final_name=intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum-production:2.5.0-SNAPSHOT-build
while getopts ":c:i:f:" opt
do
    case $opt in
        c)
        export container_name=$OPTARG
        echo "container_name:" $container_name
        ;;
        i)
        export image_name=$OPTARG
        echo "image_name:" $image_name
        ;;
        f)
        export final_name=$OPTARG
        echo "final_name:" $final_name
        ;;
        ?)
        echo $OPTARG
        echo "unknow parameter"
        exit 1;;
    esac
done
#Clean up old
sudo docker rm -f $container_name

# Run new command in container
sudo docker run -i \
        --net=host \
        --name=$container_name \
        -e SGX_MEM_SIZE=15GB \
        -e SGX_THREAD=2048 \
        -e SGX_HEAP=1GB \
        -e SGX_KERNEL_HEAP=1GB \
        -e ENABLE_SGX_DEBUG=true \
        -e ATTESTATION=true \
        -e USING_TMP_HOSTFS=false \
        $image_name \
        bash /opt/run_spark_on_occlum_glibc.sh init
echo "build finish"
docker commit $container_name $final_name
#clean up
sudo docker rm -f $container_name

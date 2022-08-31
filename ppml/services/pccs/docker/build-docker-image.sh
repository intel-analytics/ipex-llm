export http_proxy=your_http_proxy
export https_proxy=your_https_proxy
export PCCS_IMAGE_NAME=your_pccs_image_name_to_build
export PCCS_IMAGE_VERSION=your_pccs_image_version_to_build

sudo docker build \
    --build-arg http_proxy=${http_proxy} \
    --build-arg https_proxy=${https_proxy} \
    -t $PCCS_IMAGE_NAME:$PCCS_IMAGE_VERSION -f ./Dockerfile .

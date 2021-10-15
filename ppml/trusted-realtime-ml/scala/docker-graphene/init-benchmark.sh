cd ../models/resnet50
mkdir fp32 int8
mv resnet_v1_50.* fp32
wget -c "https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/openvino/2018_R5/resnet_v1_50_i8.bin/download" -O int8/resnet_v1_50_i8.bin && \
wget -c "https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/openvino/2018_R5/resnet_v1_50_i8.xml/download" -O int8/resnet_v1_50_i8.xml

sed -i 's/resnet50/resnet50\/fp32/g' ../../../config.yaml

cd ../../benchmark
apt install -y libgl1-mesa-glx python3-pip
pip3 install --upgrade pip
pip3 install --pre --upgrade analytics-zoo
pip3 install -r requirement.yml

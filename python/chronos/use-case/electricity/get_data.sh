if ! [ -d "data" ]; then 
    mkdir data; 
    cd data; 
    wget https://cloud.tsinghua.edu.cn/seafhttp/files/20aed153-7deb-45ff-8faf-9dd3e609fe97/electricity.csv;
fi

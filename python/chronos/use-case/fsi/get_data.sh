if ! [ -d "data" ]; then 
    mkdir data; 
    cd data; 
    wget https://github.com/CNuge/kaggle-code/raw/master/stock_data/individual_stocks_5yr.zip;
    wget https://raw.githubusercontent.com/CNuge/kaggle-code/master/stock_data/merge.sh;
    chmod +x merge.sh;
    unzip individual_stocks_5yr.zip;
    ./merge.sh;
fi

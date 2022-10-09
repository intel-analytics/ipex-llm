conda create -y -n chronos python=3.7 setuptools=58.0.4
source activate chronos
pip install --no-cache-dir neural_compressor==1.8.1 && \
pip install --no-cache-dir onnxruntime==1.6.0 && \
pip install --no-cache-dir tsfresh==0.17.0 && \
pip install --no-cache-dir numpy==1.19.5 && \
pip install --no-cache-dir ray==1.9.2 ray[tune]==1.9.2 ray[default]==1.9.2 && \
pip install --no-cache-dir pyarrow==6.0.1 && \
pip install --no-cache-dir --pre bigdl-nano[pytorch] && \
pip install --no-cache-dir --pre bigdl-nano[tensorflow] && \
pip install --no-cache-dir torchmetrics==0.7.2 && \
pip install --no-cache-dir scipy==1.5.4 && \
pip install --no-cache-dir prometheus_pandas==0.3.1 && \
pip install --no-cache-dir xgboost==1.2.0 && \
pip install --no-cache-dir jupyter==1.0.0

model=$1
auto_tuning=$2
hardware=$3
options=()

if [ $model == "pytorch" ] || [ $model == "tensorflow" ];
then
    options[0]=$model

    if [ $auto_tuning == "y" ];
    then
        options[1]="automl"
    elif [ $auto_tuning != "n" ];
    then
        # invalid args
        echo "Invalid argument."
        echo "Argument auto_tuning can be y (for yes)|n (for no), please check."
        exit -1
    fi

    if [ $hardware == "cluster" ];
    then
        options[2]="distributed"
    elif [ $hardware != "single" ];
    then
        # invalid args
        echo "Invalid argument."
        echo "Argument hardware can be single|cluster, please check."
        exit -1
    fi
elif [ $model == "prophet" ] || [ $model == "arima" ];
then
    if [ $auto_tuning == "y" ];
    then
        options[0]="distributed"
    elif [ $auto_tuning != "n" ];
    then
        # invalid args
        echo "Invalid argument."
        echo "Argument auto_tuning can be y (for yes)|n (for no), please check."
        exit -1
    fi

    if [ $hardware == "cluster" ];
    then
        if [ ${#options[@]} == 0 ];
        then
            options[0]="distributed"
        fi
    elif [ $hardware != "single" ];
    then
        # invalid args
        echo "Invalid argument."
        echo "Argument hardware can be single|cluster, please check."
        exit -1
    fi

    if [ $model == "prophet" ];
    then
        pip install --no-cache-dir prophet==1.1.0
    else
        # ARIMA
        pip install --no-cache-dir pmdarima==1.8.5
    fi
else
    # invalid args
    echo "Invalid argument."
    echo "Argument model can be pytorch|tensorflow|prophet|arima, please check."
    exit -1
fi

if [ ${#options[@]} == 0 ];
then
    pip install --no-cache-dir --pre --upgrade bigdl-chronos
else
    echo $options
    str=${options[*]}
    opts=`echo $str | tr ' ' ','`
    echo $opts
    pip install --no-cache-dir --pre --upgrade bigdl-chronos[$opts]
fi




if (( $# < 4)); then
    echo "Usage: install-python-env.sh model auto_tuning hardware extra_dep"
    echo "       model: pytorch|tensorflow|prophet|arima"
    echo "       auto_tuning: y|n"
    echo "       hardware: single|cluster"
    echo "       extra_dep: y|n"
    exit -1
fi

conda create -y -n chronos python=3.7 setuptools=58.0.4
source activate chronos

model=$1
auto_tuning=$2
hardware=$3
extra_dep=$4
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
        echo "Argument auto_tuning can be y (for yes) or n (for no), please check."
        exit -1
    fi

    if [ $hardware == "cluster" ];
    then
        options[2]="distributed"
    elif [ $hardware != "single" ];
    then
        # invalid args
        echo "Invalid argument."
        echo "Argument hardware can be single or cluster, please check."
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
        echo "Argument auto_tuning can be y (for yes) or n (for no), please check."
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
        echo "Argument hardware can be single or cluster, please check."
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
    echo "Argument model can be pytorch, tensorflow, prophet, or arima, please check."
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

if [ $extra_dep == "y" ];
then
    pip install --no-cache-dir neural_compressor==1.8.1 && \
    pip install --no-cache-dir onnxruntime==1.6.0 && \
    pip install --no-cache-dir onnx==1.8.0 && \
    pip install --no-cache-dir tsfresh==0.17.0 && \

    pip install --no-cache-dir prometheus_pandas==0.3.1 && \
    pip install --no-cache-dir xgboost==1.2.0 && \
    pip install --no-cache-dir jupyter==1.0.0
elif [ $extra_dep != "n" ];
then
    # invalid args
    echo "Invalid argument."
    echo "Argument extra_dep can be y (for yes) or n (for no), please check."
    exit -1
fi



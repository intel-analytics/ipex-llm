conda create -y -n chronos python=3.7 setuptools=58.0.4
source activate chronos

model=$1
auto_tuning=$2
hardware=$3
inference=$4
extra_dep=$5
options=()

if [ $model == "pytorch" ] || [ $model == "tensorflow" ];
then
    options+=($model)

    if [ $auto_tuning == "y" ];
    then
        options+=("automl")
    elif [ $auto_tuning != "n" ];
    then
        # invalid args
        echo "Invalid argument."
        echo "Argument auto_tuning can be y (for yes) or n (for no), please check."
        exit -1
    fi

    if [ $hardware == "cluster" ];
    then
        options+=("distributed")
    elif [ $hardware != "single" ];
    then
        # invalid args
        echo "Invalid argument."
        echo "Argument hardware can be single or cluster, please check."
        exit -1
    fi

    if [ $inference == "y" ];
    then
        options+=("inference")
    elif [ $inference != "n" ];
    then
        # invalid args
        echo "Invalid argument."
        echo "Argument inference can be y or n, please check."
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
elif [ $model != "ml" ];
then
    # invalid args
    echo "Invalid argument."
    echo "Argument model can be pytorch, tensorflow, prophet, arima or ml, please check."
    exit -1
fi

if [ ${#options[@]} == 0 ];
then
    pip install --no-cache-dir --pre --upgrade bigdl-chronos
else
    str=${options[*]}
    opts=`echo $str | tr ' ' ','`
    pip install --no-cache-dir --pre --upgrade bigdl-chronos[$opts]
fi

if [ $extra_dep == "y" ];
then
    pip install --no-cache-dir tsfresh==0.17.0 && \
    pip install --no-cache-dir pyarrow==6.0.1 && \

    pip install --no-cache-dir prometheus_pandas==0.3.1 && \
    pip install --no-cache-dir xgboost==1.2.0 && \
    pip install --no-cache-dir jupyter==1.0.0 && \
    pip install --no-cache-dir matplotlib
elif [ $extra_dep != "n" ];
then
    # invalid args
    echo "Invalid argument."
    echo "Argument extra_dep can be y (for yes) or n (for no), please check."
    exit -1
fi

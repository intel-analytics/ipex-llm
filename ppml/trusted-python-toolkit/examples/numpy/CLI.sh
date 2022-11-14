num=4096
env='native'
dtype='int'

while getopts "n:p:t:" opt
do
    case $opt in
        n)
            num=$OPTARG
        ;;
        p)
            env=$OPTARG
        ;;
        t)
            dtype=$OPTARG
    esac
done

python /ppml/examples/numpy/benchmark-numpy.py --size=$num --env=$env --type=$dtype


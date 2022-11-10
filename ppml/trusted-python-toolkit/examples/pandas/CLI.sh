num=10
env='native'

while getopts "n:p:" opt
do
    case $opt in
        n)
            num=$OPTARG
        ;;
        p)
            env=$OPTARG
        ;;
    esac
done

python /ppml/examples/pandas/benchmark-pandas.py --size=$num --env=$env


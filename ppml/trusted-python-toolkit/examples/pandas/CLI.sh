dataset='/ppml/work/data/dataset.csv'
env='native'

while getopts "d:p:" opt
do
    case $opt in
        d)
            dataset=$OPTARG
        ;;
        p)
            env=$OPTARG
        ;;
    esac
done

python /ppml/examples/pandas/benchmark-pandas.py --dataset=$dataset --env=$env


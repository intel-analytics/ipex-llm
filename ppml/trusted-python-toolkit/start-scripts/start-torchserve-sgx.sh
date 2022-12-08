configFile=""
while getopts "c:" opt
do
    case $opt in
        c)
            configFile=$OPTARG
        ;;
    esac
done
cd /ppml
./init.sh
cd /ppml/work/data
cat $configFile | while read line
do
        if [[ $line =~ "minWorkers" ]]
        then
                line=${line#*\"minWorkers\": }
                line=${line%%,*}

                port=9000
                for ((i=0;i<line;i++,port++))
                do
                (
                        bash /ppml/work/start-scripts/start-backend-sgx.sh -p $port
                )&
                done
        fi
done
(
        bash /ppml/work/start-scripts/start-frontend-sgx.sh -c $configFile
)&
wait


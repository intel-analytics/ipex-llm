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
port=9000
cat $configFile | while read line
do
        while [[ $line =~ "minWorkers" ]]
        do
                line=${line#*\"minWorkers\": }
                num=${line%%,*}
                line=${line#*,}

                for ((i=0;i<num;i++,port++))
                do
                (
                        bash /ppml/work/start-scripts/start-backend-sgx.sh -p $port
                )&
                done
        done
done
(
        bash /ppml/work/start-scripts/start-frontend-sgx.sh -c $configFile
)&
wait


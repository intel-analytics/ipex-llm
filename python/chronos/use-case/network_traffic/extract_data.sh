#! /bin/bash

cd data
echo "StartTime,EndTime,AvgRate,total" > data.csv
for year in "2018" "2019"; do
	for i in $(seq 1 1 12)
	do
		month=$year$(printf %02d $i)
		file_name=${month}.agr

                cmd1="grep 'StartTime' $file_name | sed 's/.*(\(.*\))/\1/'"
                cmd2="grep 'EndTime' $file_name | sed 's/.*(\(.*\))/\1/'"
                cmd3="grep 'AvgRate' $file_name | awk '{print \$2}'"
                cmd4="grep 'total' $file_name | awk '{print \$2}'"
                paste -d ',' <(eval $cmd1) <(eval $cmd2) <(eval $cmd3) <(eval $cmd4) >> data.csv 
			
	done
done

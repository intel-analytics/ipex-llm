#! /bin/bash

mkdir "data"
cd "data"

for year in "2018" "2019"; do
	for i in $(seq 1 1 12)
	do
		month=$year$(printf %02d $i)
		echo $month
		curl -O http://mawi.wide.ad.jp/~agurim/dataset/$month/${month}.agr
	done
done

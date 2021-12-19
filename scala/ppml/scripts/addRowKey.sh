#!bin/bash
awk '{if (NR==1) print "'$1',"$0; else print (NR-1)","$0}' $2 | tee $2

#!/usr/bin/env bash

if [ ! -z "$1" ]
then
   DIR=$1
   cd "$DIR"
else
   DIR=$(dirname "$0")
   echo "Download path: $DIR"
   cd "$DIR"
fi

FILENAME="./m_1932.csv"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
   exit
fi

FILENAME="./machine_usage.csv"
if [ ! -f "$FILENAME" ]
then
   echo "$FILENAME doesn't exists."
   bash ./get_data.sh
fi

echo "Extract m_1932"
grep m_1932 machine_usage.csv > m_1932.csv

echo "Finished"

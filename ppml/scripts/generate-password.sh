#!/bin/bash
OUTPUT=path_of_output_yaml_file
PASSWORD_PATH=./password

mkdir -p password && cd password
export PASSWORD=$1 #used_password_when_generate_keys
openssl genrsa -out key.txt 2048
echo $PASSWORD | openssl rsautl -inkey key.txt -encrypt >output.bin

echo apiVersion: v1 > $OUTPUT
echo data: >> $OUTPUT
echo " key.txt:" >> $OUTPUT
echo -n "  " >> $OUTPUT
base64 -w 0 $PASSWORD_PATH/key.txt  >> $OUTPUT
echo "" >> $OUTPUT
echo " output.bin:" >> $OUTPUT
echo -n "  " >> $OUTPUT
base64 -w 0 $PASSWORD_PATH/output.bin  >> $OUTPUT
echo "" >> $OUTPUT
echo kind: Secret  >> $OUTPUT
echo metadata:  >> $OUTPUT
echo " name: ssl-password"  >> $OUTPUT
echo type: Opaque   >> $OUTPUT

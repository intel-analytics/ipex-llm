#!/usr/bin/env bash

times=$1
[ "$times" -gt 0 ] 2>/dev/null || times=300

for (( c=1; c<=times; c++ ))
do
	echo "hello $c"
	cp somefile testdir/somefile
	cat somefile > testdir/createdfile
	ls testdir/
	rm -rf testdir/somefile testdir/createdfile
	date +"current date is %D"
done

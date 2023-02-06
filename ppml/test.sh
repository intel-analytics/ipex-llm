#!/bin/bash

files=$(find . -type f)

IFS=$'\n'; set -f

old_pattern='2\.1\.0-SNAPSHOT'

new_pattern='2\.3\.0-SNAPSHOT'

for fname in $files; do
  grep -q $old_pattern "$fname" &&
  nvim -c "%s/$old_pattern/$new_pattern/gc" -c 'wq' "$fname"
done

unset +f

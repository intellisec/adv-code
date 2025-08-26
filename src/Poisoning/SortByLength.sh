#!/bin/bash

# sort a text file by line length (descending)
if [[ $# -ne 2 ]]
then
    echo "Usage: $0 <src> <dest>"
    exit
fi

cat "$1" | awk '{print length, $0}' | sort -nsr | cut -d" " -f2- > "$2"

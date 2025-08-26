#!/bin/bash

# Running this should for a specific grepstring should replicate
# the result of CommonSubstrings.py for that specific string
#
# Example:
# python Poisoning/CommonSubstrings.py --file Poisoning/common/searchresults_100k_sorted.txt -k 10000 -m 35 --limit-lines 50 -w --o /tmp/substringsearch.txt
# may yield "11024" for a string "law or agreed to in writing, software"
#
# Then running this script as
# Poisoning/SanityCheck.sh "Poisoning/common/searchresults_100k_sorted.txt" 50 "law or agreed to in writing, software"
# should also print 11024
if [[ $# -ne 3 ]]
then
    echo "Usage: $0 <inputfile> <linecount> <grepstring>"
    exit
fi
head -$2 "$1" | grep "$3" | awk '{ sum += $1} END { print sum }'

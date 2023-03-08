#!/bin/bash


AGENT='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'
DIR=raw
mkdir $DIR
wget -U "$AGENT" -E -H -k -K -p -e robots=off -P $DIR -i ./urls

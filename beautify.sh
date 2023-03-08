#!/bin/bash

INPUT=`readlink -e filter`
OUTPUT=smooth
mkdir -p $OUTPUT
OUTPUT=`readlink -e $OUTPUT`
cd ../GFPGAN
. venv/bin/activate
python3 inference_gfpgan.py -i "$INPUT" -o "$OUTPUT" -v 1.3 -s 2



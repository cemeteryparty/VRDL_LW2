#!/bin/bash

if [ "$1" = "dbg" ]; then
	retinanet-debug --show-annotations --image-min-side 100 --image-max-side 220 \
		--no-gui --output-dir debug_tra csv tra_annotations.csv classes.csv
	retinanet-debug --show-annotations --image-min-side 100 --image-max-side 220 \
		--no-gui --output-dir debug_val csv val_annotations.csv classes.csv
	ls -l debug_tra/train | wc -l
	ls -l debug_val/train | wc -l
elif [ "$1" = "clean" ]; then
	rm -rf debug_tra debug_val
else
	echo "E: ./debug.sh [dbg | clean]"
fi


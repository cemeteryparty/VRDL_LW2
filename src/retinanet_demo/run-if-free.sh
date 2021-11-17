#!/bin/bash

while true; do
	occ0=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | wc -l)
	occ1=$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader | wc -l)
	printf "${occ0} ${occ1}\r"
	if [ "${occ0}" -eq "0" ]; then
		echo "GPU 0 is free now      "
		export CUDA_VISIBLE_DEVICES="0"
		break
	elif [ "${occ0}" -eq "0"]; then
		echo "GPU 1 is free now      "
		export CUDA_VISIBLE_DEVICES="1"
		break
	fi
	printf "GPU 0 has ${occ0} proc, GPU 1 has ${occ1} proc."
	sleep 5
done

wget https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5
python demo.py


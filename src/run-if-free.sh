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

# --compute-val-loss --snapshot  
#retinanet-train --snapshot resnet50_csv_01.h5 --compute-val-loss --initial-epoch 1 \
retinanet-train --batch-size 32 --epochs 50 --steps 200 \
	--snapshot-path /mnt/left/phlai_DATA/vrdl5008/models/ \
	--image-min-side 100 --image-max-side 220 --compute-val-loss \
	csv tra_annotations.csv classes.csv --val-annotations val_annotations.csv > \
	output/train.log

exit 0
for ((i=10; i<=30; i+=10)) do
	retinanet-convert-model /mnt/left/phlai_DATA/vrdl5008/models/resnet50_csv_${i}.h5 \
		/mnt/left/phlai_DATA/vrdl5008/minf/resnet50_ep${i}_1116.h5
	retinanet-evaluate --image-min-side 100 --image-max-side 220  \
		csv val_annotations.csv classes.csv \
		/mnt/left/phlai_DATA/vrdl5008/minf/resnet50_ep${i}_1116.h5 > \
		output/resnet50_ep${i}_1116.log
done

exit 0
python gen_Anno.py
./debug.sh dbg


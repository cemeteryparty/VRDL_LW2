# VRDL LabWork002

# Top_Reproduce `answer.json` (Using inference.py)

Download models and `inference.ipynb`, `inference.py` from below link first.

gdrive resource link: https://drive.google.com/drive/folders/12iLYk5nN1OkXz0EMGODHN_4LNSOsJ6do?usp=sharing

## GDrive Structure

 - resource
 	- `all_inf_ep50.h5`
 	- `inference.py (generate answer.json)`
 	- `inference.ipynb (estimate inference time)`
 	- test.zip (Please download from 2021 VRDL HW2 Competition)

Generate `answer.json`, checking the directory structure before executing
```sh
$ unzip -qq test.zip -d ./
$ rm -rf __MACOSX/
$ python3 inference.py --image-min-side 150 --image-max-side 330 --model-path all_inf_ep50.h5
$
```

# Instruction

## Preparation

```sh
# Download train.zip and test.zip from 2021 VRDL HW2 Competition First
$ cd src
$ make
$ make install # install keras-retinanet
```

## Generate Annotation

```sh
$ python3 gen_Anno.py
```

## Training Process

```sh
# First training
$ python3 train.py --batch-size <Batch Size> --epochs <Epoch> --steps <Step Per Epoch> \
> --lr <Learning Rate> --image-min-side 150 --image-max-side 330 --compute-val-loss \
> --snapshot-path <Save Model Path> --random-transform \
> csv tra_annotations.csv classes.csv --val-annotations val_annotations.csv

# From pre-trained model
$ python3 train.py --snapshot <Pre-Trained Model Path> \
> --batch-size <Batch Size> --epochs <Epoch> --steps <Step Per Epoch> \
> --lr <Learning Rate> --image-min-side 150 --image-max-side 330 --compute-val-loss \
> --snapshot-path <Save Model Path> --random-transform \
> csv tra_annotations.csv classes.csv --val-annotations val_annotations.csv
```

## Testing

```sh
# convert Classificaion-Regeression Model to Inference Model
$ retinanet-convert-model <Save Model Path>/resnet50_csv_<Epoch_NUM>.h5 \
> <Save Inference Model Path>
$ python3 inference.py --image-min-side 150 --image-max-side 330 --model-path <Save Inference Model Path>
```

all:
	unzip /mnt/left/phlai_DATA/vrdl5008/train.zip -d ./
	cp /mnt/left/phlai_DATA/vrdl5008/digitStruct.mat ./
	rm -rf __MACOSX/ train/see_bboxes.m train/digitStruct.mat
install:
	pip install keras-retinanet
#	git clone https://github.com/fizyr/keras-retinanet.git
#	git clone https://github.com/broadinstitute/keras-resnet.git
#build: keras-retinanet
#	cd keras-retinanet; pwd; pip install . --user; python setup.py build_ext --inplace
uninstall:
	rm -rf keras-retinanet keras-resnet
clean:
	rm -r train
	rm digitStruct.mat *.csv


all:
	unzip train.zip -d ./
	cp train/digitStruct.mat ./
	rm -rf __MACOSX/ train/see_bboxes.m train/digitStruct.mat
	unzip -qq test.zip -d ./
	rm -rf __MACOSX/
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


BUILD_DIR=build
PKG_DIR=${BUILD_DIR}/python

default: build zip

clean:
	rm *.zip
	rm -rf ${BUILD_DIR} && mkdir -p ${PKG_DIR}

build: clean
	docker run --rm -v $(shell pwd):/foo -w /foo lambci/lambda:build-python3.7 \
		pip install -r requirements.txt -t ${PKG_DIR}

zip:
	zip -r drapegnik-Python37-numpy-aws-x-ray-sdk.zip python

install:
	pip install -r requirements.txt

test:
	python test.py

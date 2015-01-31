.PHONY: all clean build test install uninstall

all: clean build test

clean:
	rm lavaburst/*.so

build:
	python setup.py build_ext --inplace

test:
	nosetests

install:
	pip install -e .

uninstall:
	pip uninstall lavaburst

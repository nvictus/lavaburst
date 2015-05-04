BUILDDIR=lavaburst/core

.PHONY: all clean docs build test install uninstall

all: clean build test

docs:
	cd docs && $(MAKE) html

clean:
	rm $(BUILDDIR)/*.so

build:
	python setup.py build_ext --inplace

test:
	nosetests

install:
	pip install -e .

uninstall:
	pip uninstall lavaburst

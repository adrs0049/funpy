#
# Makefile to produce python bindings for rowmap
#

V = 0
python_exec=python3

CWD := $(shell pwd)

SOURCES=ufuncs.pyx detail.pyx detail.pxd

.DELETE_ON_ERROR:

all: cythonize

.refresh:
	touch .refresh

cythonize: $(SOURCES)
	$(python_exec) setup.py build_ext --inplace

.PHONY: setup clean force all

clean:
	rm -f *.so

# Development commands and the local pybind11 BPE build.
SHELL := /bin/sh
.DEFAULT_GOAL := help

CXX ?= c++
PYTHON ?= .venv/bin/python

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    PLATFORM_FLAGS := -undefined dynamic_lookup
else
    PLATFORM_FLAGS :=
endif

CXXFLAGS ?= -O3 -Wall -Wextra -Wno-unused-parameter
NATIVE_FLAGS := -shared -std=c++17 -fPIC $(PLATFORM_FLAGS)
PYBIND_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

NATIVE_DIR := cs336_basics/tokenization/_native
PACKAGE_DIR := cs336_basics/tokenization
NATIVE_SOURCE := $(NATIVE_DIR)/bpe_native.cpp
NATIVE_TARGET := $(PACKAGE_DIR)/_bpe_native$(EXT_SUFFIX)

native: $(NATIVE_TARGET)

$(NATIVE_TARGET): $(NATIVE_SOURCE) Makefile
	@mkdir -p $(PACKAGE_DIR)
	$(CXX) $(CXXFLAGS) $(NATIVE_FLAGS) $(PYBIND_INCLUDES) $(NATIVE_SOURCE) -o $(NATIVE_TARGET)

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check cs336_basics tests

typecheck:
	$(PYTHON) -m ty check cs336_basics

check: lint typecheck test

test-native-batch: native
	$(PYTHON) -m pytest tests/test_tokenizer.py -k native_batch -v

test-cached-native: native
	$(PYTHON) -m pytest tests/test_tokenizer.py -k cached_native -v

benchmark-tokenizers: native
	$(PYTHON) -m cs336_basics.tokenization.benchmark

clean-native:
	rm -f $(PACKAGE_DIR)/_bpe_native*.so \
		$(PACKAGE_DIR)/_bpe_native*.dylib \
		$(PACKAGE_DIR)/_bpe_native*.pyd

# Remove bytecode only from project-owned Python trees; never traverse .venv.
clean-pyc:
	find cs336_basics tests -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	find cs336_basics tests -depth -type d -name '__pycache__' -empty -delete

clean-cache: clean-pyc
	rm -rf .pytest_cache .ruff_cache .mypy_cache .ty_cache

clean: clean-native clean-cache

help:
	@echo "Available targets:"
	@echo "  native              Build the optional native BPE extension"
	@echo "  test                 Run the complete test suite"
	@echo "  lint                 Run Ruff"
	@echo "  typecheck            Run ty"
	@echo "  check                Run lint, type checking, and tests"
	@echo "  clean-pyc            Remove project .pyc/.pyo and empty __pycache__ directories"
	@echo "  clean-cache          Also remove test/linter/type-checker caches"
	@echo "  clean-native         Remove the compiled native extension"
	@echo "  clean                Remove native and Python-generated artifacts"

.PHONY: benchmark-tokenizers check clean clean-cache clean-native clean-pyc help lint native test \
	test-cached-native test-native-batch typecheck

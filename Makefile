# Local pybind11 build for the native BPE tokenizer.
CXX := c++
PYTHON ?= .venv/bin/python

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    PLATFORM_FLAGS := -undefined dynamic_lookup
else
    PLATFORM_FLAGS :=
endif

CXXFLAGS := -O3 -Wall -Wextra -Wno-unused-parameter -shared -std=c++17 -fPIC $(PLATFORM_FLAGS)
PYBIND_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

NATIVE_DIR := cs336_basics/tokenization/_native
PACKAGE_DIR := cs336_basics/tokenization
NATIVE_SOURCE := $(NATIVE_DIR)/bpe_native.cpp
NATIVE_TARGET := $(PACKAGE_DIR)/_bpe_native$(EXT_SUFFIX)

native: $(NATIVE_TARGET)

$(NATIVE_TARGET): $(NATIVE_SOURCE)
	$(CXX) $(CXXFLAGS) $(PYBIND_INCLUDES) $(NATIVE_SOURCE) -o $(NATIVE_TARGET)

test-native-batch: native
	$(PYTHON) -m pytest tests/test_tokenizer.py -k native_batch -v

test-cached-native: native
	$(PYTHON) -m pytest tests/test_tokenizer.py -k cached_native -v

benchmark-tokenizers: native
	$(PYTHON) -m cs336_basics.tokenization.benchmark

clean-native:
	rm -f $(PACKAGE_DIR)/_bpe_native*.so $(PACKAGE_DIR)/_bpe_native*.dylib

.PHONY: native test-native-batch test-cached-native benchmark-tokenizers clean-native

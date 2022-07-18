# Disable implicit rules so that Make doesn't try to compile the binaries
.SUFFIXES:

.PHONY: all
all:
	@echo "Use gramine-test for running tests."

# TODO: Python tests should not produce leftovers
.PHONY: clean
clean:
	gramine-test clean
	$(RM) -r \
		.pytest_cache \
		__pycache__ \
		*.xml \
		tmp/*

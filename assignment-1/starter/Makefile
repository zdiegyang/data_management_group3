.DEFAULT_GOAL := help

PYTHON ?= python

.PHONY: help check inventory run train trace test

help:
	@echo "Student pipeline commands"
	@echo ""
	@echo "  make check      Verify object-store and PostgreSQL connections"
	@echo "  make inventory  List the supplied course data files"
	@echo "  make run        Run your pipeline after implementing the stages"
	@echo "  make train      Train/evaluate models from the required ML input tables"
	@echo "  make trace      Trace one syndrome and one Google prediction back to Bronze (after make train)"
	@echo "  make test       Run automated tests"

check:
	$(PYTHON) -m quantum_lake_student.cli check

inventory:
	$(PYTHON) -m quantum_lake_student.cli inventory

run:
	$(PYTHON) -m quantum_lake_student.cli run

train:
	$(PYTHON) -m quantum_lake_student.cli train

trace:
	$(PYTHON) -m quantum_lake_student.cli trace

test:
	$(PYTHON) -m pytest

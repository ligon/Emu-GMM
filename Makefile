POETRY = poetry

FILES ?=

ifeq ($(strip $(FILES)),)
RUFF_TARGET = .
BLACK_TARGET = .
MYPY_TARGET = src tests
PYTEST_TARGET =
PYTEST_FLAGS = -m "not slow"
else
RUFF_TARGET = $(FILES)
BLACK_TARGET = $(FILES)
MYPY_TARGET = $(FILES)
PYTEST_TARGET = $(FILES)
PYTEST_FLAGS =
endif

ifdef PYTEST_TARGET
PYTEST_CMD = $(POETRY) run pytest $(PYTEST_TARGET)
else
PYTEST_CMD = $(POETRY) run pytest $(PYTEST_FLAGS)
endif

.PHONY: setup lint black mypy test test-parallel check quick-check slow-tests clean build publish release

setup: .venv/pyvenv.cfg

.venv/pyvenv.cfg: pyproject.toml
	$(POETRY) install
	@touch $@

lint:
	$(POETRY) run ruff check $(RUFF_TARGET)

black:
	$(POETRY) run black --check $(BLACK_TARGET)

mypy:
	$(POETRY) run mypy $(MYPY_TARGET)

test:
	$(PYTEST_CMD)

test-parallel:
	$(POETRY) run pytest -n auto $(PYTEST_FLAGS) $(PYTEST_TARGET)

quick-check: lint black mypy
	$(POETRY) run pytest -m "not slow" $(PYTEST_TARGET)

check: lint black mypy
	$(POETRY) run pytest $(PYTEST_TARGET)

slow-tests:
	$(POETRY) run pytest -m slow $(PYTEST_TARGET)

clean:
	rm -rf build dist *.egg-info
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

build:
	$(POETRY) build

publish:
	$(POETRY) publish

BUMP ?= patch
release: build
	$(eval NEW_VER := $(shell $(POETRY) version $(BUMP) -s))
	git add pyproject.toml
	git commit -m "Bump version to $(NEW_VER)"
	git tag v$(NEW_VER)
	@echo "Tagged v$(NEW_VER)."

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

# Usage: make release BUMP=patch   (or minor, major, prepatch, ...)
#
# Order matters (fixed 2026-06-09): gate on the full check, require a
# clean tree, bump the version FIRST, then commit + annotated tag, and
# only then build -- so the dist/ artifacts carry the released version.
# (The previous target built before bumping, leaving a stale-version
# artifact under the new tag; the same latent flaw exists in
# ../ManifoldGMM's Makefile, from which this target was adapted.)
# Tags are local to this mirror: pull them from the source repo with
# `git fetch coder --tags`. `make publish` remains a separate, explicit
# step.
BUMP ?= patch
release: check
	@git diff --quiet && git diff --cached --quiet || \
		{ echo "release: working tree not clean; commit or stash first"; exit 1; }
	$(eval NEW_VER := $(shell $(POETRY) version $(BUMP) -s))
	git add pyproject.toml
	git commit -m "Bump version to $(NEW_VER)"
	git tag -a v$(NEW_VER) -m "emu-gmm v$(NEW_VER)"
	$(POETRY) build
	@echo "Tagged v$(NEW_VER) and built dist/ at that version."
	@echo "Publish (if desired) with 'make publish'; fetch the tag from"
	@echo "the source repo with 'git fetch coder --tags'."

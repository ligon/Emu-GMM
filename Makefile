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
PYTEST_CMD = $(POETRY) run python -m pytest $(PYTEST_TARGET)
else
PYTEST_CMD = $(POETRY) run python -m pytest $(PYTEST_FLAGS)
endif

.PHONY: setup lint black mypy test test-parallel check quick-check slow-tests clean build publish release

setup: .venv/pyvenv.cfg

.venv/pyvenv.cfg: pyproject.toml
	$(POETRY) install
	@touch $@

lint:
	$(POETRY) run python -m ruff check $(RUFF_TARGET)

black:
	$(POETRY) run python -m black --check $(BLACK_TARGET)

mypy:
	$(POETRY) run python -m mypy $(MYPY_TARGET)

test:
	$(PYTEST_CMD)

test-parallel:
	$(POETRY) run python -m pytest -n auto $(PYTEST_FLAGS) $(PYTEST_TARGET)

quick-check: lint black mypy
	$(POETRY) run python -m pytest -m "not slow" $(PYTEST_TARGET)

check: lint black mypy
	$(POETRY) run python -m pytest $(PYTEST_TARGET)

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
# The pre-release gate defaults to the full `check`. When a full check
# has JUST passed on the identical tree, skip the redundant ~30-minute
# re-run with `make release GATE=` (empty gate). The clean-tree guard
# still applies either way.
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
# NOTE: the bump runs as a recipe SHELL line, not $(eval $(shell ...)).
# GNU make expands an entire recipe (including $(shell ...)) before
# executing its first line, so an eval-style bump fires BEFORE the
# clean-tree guard and trips it on its own modification. The same
# landmine exists in ../ManifoldGMM's release target.
BUMP ?= patch
GATE ?= check
release: $(GATE)
	@git diff --quiet && git diff --cached --quiet || \
		{ echo "release: working tree not clean; commit or stash first"; exit 1; }
	@NEW_VER=$$($(POETRY) version $(BUMP) -s) && \
	git add pyproject.toml && \
	git commit -m "Bump version to $$NEW_VER" && \
	git tag -a v$$NEW_VER -m "emu-gmm v$$NEW_VER" && \
	$(POETRY) build && \
	echo "Tagged v$$NEW_VER and built dist/ at that version." && \
	echo "Publish (if desired) with 'make publish'; fetch the tag from" && \
	echo "the source repo with 'git fetch coder --tags'."

install:
		uv sync

run:
	uv run python -m src $(filter-out run,$(MAKECMDGOALS))

debug:
	uv run python -m pdb src $(filter-out run,$(MAKECMDGOALS))

%:
	@:

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache

lint:
	flake8 src
	mypy src --warn-return-any \
	--warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs \
	--check-untyped-defs
To build the docs locally run:

```bash
# Install the docs dependencies (from the repo root)
uv sync --extra docs

# Link the tutorials folder to the current directory
ln -s ../tutorials tutorials

# Build the docs
uv run sphinx-apidoc -o . ../dreams && uv run make html

# Open the docs in browser
open _build/html/index.html
```

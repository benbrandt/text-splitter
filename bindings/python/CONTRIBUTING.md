# Python binding development

Run the following commands from `bindings/python`.

## Type stubs

`semantic_text_splitter.pyi` is generated from the Rust bindings and their
documentation. Do not edit it by hand.

```sh
python -m pip install -r requirements-typing.txt
python scripts/generate_stubs.py
python scripts/generate_stubs.py --check
mypy --strict --python-version 3.10 tests/typecheck.py
```

The generator uses the default `abi3-py310` feature so that the output targets
the oldest supported Python, even when a newer interpreter runs the command.
Do not enable `abi3t` or disable default features when generating the shipped
stubs. The `stubgen` Cargo feature enables experimental PyO3 introspection only
for generation; normal wheel builds ship the checked-in file.

Keep Python-specific return annotations in `#[pyo3(signature = ...)]` where a
Rust `PyList` return value otherwise loses its element type. Custom callback
input metadata lives on `CustomCallback`'s `FromPyObject` implementation.
CI checks regeneration and both accepted and rejected API usage with mypy.
The script builds an introspectable wheel with Maturin, then uses the matching
`pyo3-introspection` release to read it. Maturin 1.15's built-in `generate-stubs`
still uses the older PyO3 0.29 introspector and misses some 0.30 metadata.
When updating PyO3 or either pinned tool, regenerate and review the stub diff,
including any changes to input protocols.

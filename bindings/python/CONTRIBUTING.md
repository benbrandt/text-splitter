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

## Python runtimes and wheels

The default build produces a GIL-enabled `cp310-abi3` wheel. It is tested on
Python 3.10, 3.14, and 3.15. Free-threaded builds retain version-specific 3.14t
wheels, and also build dual-ABI `cp315-abi3.abi3t` wheels with:

```sh
maturin build --release --locked --no-default-features --features abi3t -i python3.15t
```

The dual-ABI wheel is tested on both ordinary 3.15 and free-threaded 3.15t.
Runtime jobs install each wheel explicitly so that pip's preference for the
newer wheel cannot accidentally bypass coverage of the older `cp310-abi3` wheel.

CI runs the integration suite on Python 3.15 and a dependency-independent
threading suite on 3.14t/3.15t on native Linux, macOS, and Windows targets.
The threading tests check shared use of all three splitters with character,
callback, Hugging Face JSON-file, and Tiktoken sizing.
They also verify in a fresh interpreter that importing this extension does not
enable the GIL. Tests must not force `PYTHON_GIL=0`, since doing so would hide
an incorrect extension declaration.

The Python `tokenizers` and `tree-sitter-python` packages currently have build
or packaging problems on these free-threaded runtimes. Rather than patch those
dependencies or silently enable the GIL, the threading suite uses our Rust
tokenizer directly and a tiny native Rust grammar fixture. To run it locally
after installing the matching free-threaded wheel:

```sh
cargo build --locked --manifest-path tests/grammar/Cargo.toml --target-dir ../../target/test-grammar
python -m pip install pytest
python -m pytest tests/test_threading.py
```

The fixture returns a static language pointer, wrapped in the same
`tree_sitter.Language` capsule used by Python grammar packages. It is never
included in the extension or published wheels. The full third-party Python
integration suite continues to run on GIL-enabled interpreters.

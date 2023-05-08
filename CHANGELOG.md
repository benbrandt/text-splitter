# Changelog

## v0.2.0

### Breaking Changes

#### Simpler Chunking API

Simplified API for the main use case. `TextSplitter` now only exposes two chunking methods:

- `chunks`
- `chunk_indices`

The other methods are now private. It was likely that the other methods would have caused confusion since it doesn't return the semantic units themselves, but merged versions.

You also specify chunk size directly in these methods to allow reusing the `TextSplitter` for different chunk sizes.

#### Allow passing in tokenizers directly

Rather than wrapping a tokenizer in another struct, you can instead just pass a tokenizer directly into `TextSplitter::new`.

### Bug Fixes

Better handling of recursive paragraph chunking to handle when both double and single newline splits are used.

## v0.1.0

Initial release

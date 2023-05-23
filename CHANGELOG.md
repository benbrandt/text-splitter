# Changelog

## v0.3.1

### What's New

- Handle more levels of newlines. Will now find the largest newline sequence in the text, and then work back from there, treating each consecutive newline sequence length as its own semantic level.

## v0.3.0

### Breaking Changes

- Match feature names for tokenizer crates to prevent conflicts in the future.
  - `huggingface -> tokenizers`
  - `tiktoken -> tiktoken-rs`

### Features

- Moved from recursive approach to iterative approach to avoid stack overflow issues.
- Relax MSRV to 1.60.0

## v0.2.2

Add all features to docs.rs

## v0.2.1

### New Features

- impl `Default` for `TextSplitter` using `Characters`. Character count is used for chunk length by default.
- Specify the current MSRV (1.62.1)

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

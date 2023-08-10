# Changelog

## v0.4.3

### What's New

- Support `impl ChunkSizer` for `&Tokenizer` and `&CoreBPE`, allowing for generating chunks based off of a reference to a tokenizer as well, instead of requiring ownership.

## v0.4.2

### What's New

- Loosen version requirement for peer dependencies (specifically `tiktoken-rs` now supports `>=v02.0, <0.6.0`)

## v0.4.1

### What's New

- Removed unnecessary features for `tokenizers` crate to make cross-compilation easier (since tokenizer training helpers aren't needed).

## v0.4.0

### What's New

#### New Chunk Capacity (can now size chunks with Ranges)

New `ChunkCapacity` trait. When calling `splitter.chunks()` or `splitter.chunk_indices()`, the `chunk_size` argument has been replaced with `chunk_capacity`, which can be anything that implements the `ChunkCapacity` trait. This means that now the following can all be passed in:

- `usize`
- `Range<usize>`
- `RangeFrom<usize>`
- `RangeFull`
- `RangeInclusive<usize>`
- `RangeTo<usize>`
- `RangeToInclusive<usize>`

This is helpful for cases where you do have a maximum chunk size, but you don't necessarily want to fill it up all the way every time. This can be helpful in embedding cases, where you have some maximum context size, but you don't necessarily want to muddy the embeddings with lots of neighboring semantic elements. You can use a range to express this now, and the chunks will stop filling up once they have reached a size within the range.

#### Simplified Chunk Sizing traits

Simplified `ChunkSizer` trait that allows for various calculations of chunk size. No longer requires full validation logic, since that now happens within the `TextSplitter` itself.

### Breaking Changes

- `ChunkValidator` trait removed. Instead `impl ChunkSizer` instead, which just requires calculating chunk_size and not the full validation logic.
- `TokenCount` trait removed. You can just use `ChunkSizer` directly instead.
- Internal `TextChunks` iterator is no longer `pub`.

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

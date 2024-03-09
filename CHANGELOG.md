# Changelog

## v0.7.0

### What's New

**Markdown Support!** Both the Rust crate and Python package have a new `MarkdownSplitter` you can use to split markdown text. It leverages the great work of the `pulldown-cmark` crate to parse markdown according to the CommonMark spec, and allows for very fine-grained control over how to split the text.

In terms of use, the API is identical to the `TextSplitter`, so you should be able to just drop it in when you have Markdown available instead of just plain text.

#### Rust

```rust
use text_splitter::MarkdownSplitter;

// Default implementation uses character count for chunk size.
// Can also use all of the same tokenizer implementations as `TextSplitter`.
let splitter = MarkdownSplitter::default()
    // Optionally can also have the splitter trim whitespace for you. It
    // will preserve indentation if multiple lines are covered in a chunk.
    .with_trim_chunks(true);

let chunks = splitter.chunks("# Header\n\nyour document text", 1000)
```

#### Python

```python
from semantic_text_splitter import MarkdownSplitter

# Default implementation uses character count for chunk size.
# Can also use all of the same tokenizer implementations as `TextSplitter`.
# By default it will also have trim whitespace for you.
# It will preserve indentation if multiple lines are covered in a chunk.
splitter = MarkdownSplitter()
chunks = splitter.chunks("# Header\n\nyour document text", 1000)
```

### Breaking Changes

#### Rust

MSRV is now 1.75.0 since the ability to use `impl Trait` in trait methods allowed for much simpler internal APIs to enable the `MarkdownSplitter`.

#### Python

`CharacterTextSplitter`, `HuggingFaceTextSplitter`, `TiktokenTextSplitter`, and `CustomTextSplitter` classes have now all been consolidated into a single `TextSplitter` class. All of the previous use cases are still supported, you just need to instantiate the class with various class methods.

Below are the changes you need to make to your code to upgrade to v0.7.0:

##### `CharacterTextSplitter`

```python
# Before
from semantic_text_splitter import CharacterTextSplitter
splitter = CharacterTextSplitter()

# After
from semantic_text_splitter import TextSplitter
splitter = TextSplitter()
```

##### `HuggingFaceTextSplitter`

```python
# Before
from semantic_text_splitter import HuggingFaceTextSplitter
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
splitter = HuggingFaceTextSplitter(tokenizer)

# After
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
splitter = TextSplitter.from_huggingface_tokenizer(tokenizer)
```

##### `TiktokenTextSplitter`

```python
# Before
from semantic_text_splitter import TiktokenTextSplitter

splitter = TiktokenTextSplitter("gpt-3.5-turbo")

# After
from semantic_text_splitter import TextSplitter

splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo")
```

##### `CustomTextSplitter`

```python
# Before
from semantic_text_splitter import CustomTextSplitter

splitter = CustomTextSplitter(lambda text: len(text))

# After
from semantic_text_splitter import TextSplitter

splitter = TextSplitter.from_callback(lambda text: len(text))
```

## v0.6.3

- Re-release because of aggresive exclusions of benchmarks for the Rust package.

## v0.6.2

- Re-release of v0.6.1 because of wrong version tag in Python package

## v0.6.1

### Fixes

- Fix error in section filtering that didn't fix the chunk behavior regression from v0.5.0 in very tiny chunk capacities. For most commonly used chunk sizes, this shouldn't have been an issue.

## v0.6.0

### Breaking Changes

- Chunk behavior should now be the same as prior to v0.5.0. Once binary search finds the optimal chunk, we now check the next few sections as long as the chunk size doesn't change. This should result in the same behavior as before, but with the performance improvements of binary search.

## v0.5.1

### What's New

- Python bindings and Rust crate now have the same version number.

#### Rust

- Constructors for `ChunkSize` are now public, so you can more easily create your own `ChunkSize` structs for your own custom `ChunkSizer` implementation.

#### Python

- New `CustomTextSplitter` that accepts a custom callback with the signature of `(str) -> int`. Allows for custom chunk sizing on the Python side.

## v0.5.0

### What's New

- Significant performance improvements for generating chunks with the `tokenizers` or `tiktoken-rs` crates by applying binary search when attempting to find the next matching chunk size.

### Breaking Changes

- Minimum required version of `tokenizers` is now `0.15.0`
- Minimum required version of `tiktoken-rs` is now `0.5.6`
- Due to using binary search, there are some slight differences at the edges of chunks where the algorithm was a little greedier before. If two candidates would tokenize to the same amount of tokens that fit within the capacity, it will now choose the shorter text. Due to the nature of of tokenizers, this happens more often with whitespace at the end of a chunk, and rarely effects users who have set `with_trim_chunks(true)`. It is a tradeoff, but would have made the binary search code much more complicated to keep the exact same behavior.
- The `chunk_size` method on `ChunkSizer` now needs to accept a `ChunkCapacity` argument, and return a `ChunkSize` struct instead of a `usize`. This was to help support the new binary search method in chunking, and should only affect users who implemented custom `ChunkSizer`s and weren't using one of the provided ones.
  - New signature: `fn chunk_size(&self, chunk: &str, capacity: &impl ChunkCapacity) -> ChunkSize;`

## v0.4.5

### What's New

- Support `tokenizers` crate v0.15.0
- Minimum Supported Rust Version is now 1.65.0

## v0.4.4

### What's New

- Support `tokenizers` crate v0.14.0
- Minimum Supported Rust Version is now 1.61.0

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

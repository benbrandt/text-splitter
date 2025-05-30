# Changelog

## v0.27.0

### What's New

- Updated `tiktoken-rs` to v0.7.0

## v0.26.0

### What's New

- Updated to `icu` v2.0 for all unicode segmentation.
- Minimum Rust version updated to 1.82.0

## v0.25.1

### What's New

- Use `memchr` crate instead of `regex` for parsing phase in `TextSplitter`. This should improve performance in how quickly the text is parsed when scanning for newline characters.
- Implement `ChunkSizer` trait automatically for many more wrappers and references to types that already implement `ChunkSizer`.

## v0.25.0

### Breaking Changes

#### Rust

- Remove support for `rust-tokenizers` crate. This crate hasn't been updated in several years and brings in depednencies that have security warnings.

### What's New

- Use faster encoding method for `tokenizers` library, which improves performance with usage of huggingface tokenizers.

## v0.24.2

### Fixes

- Python packages target a newer version of libc to hopefully fix header file issues with tree-sitter.

### What's New

- MSRV updated to 1.81.0

## v0.24.1

### What's New

Added a new `chunk_char_indices` method to the Rust splitters.

```rust
use text_splitter::{Characters, ChunkCharIndex, TextSplitter};

let text = "\r\na̐éö̲\r\n";
let splitter = TextSplitter::new(3);
let chunks = splitter.chunk_char_indices(text).collect::<Vec<_>>();

assert_eq!(
    vec![
        ChunkCharIndex {
            chunk: "a̐é",
            byte_offset: 2,
            char_offset: 2
        },
        ChunkCharIndex {
            chunk: "ö̲",
            byte_offset: 7,
            char_offset: 5
        }
    ],
    chunks
);
```

The pulls logic from the Python bindings down into the core library. This will be more expensive than just byte offsets, and for most usage in Rust, just having byte offsets is sufficient.

However, when interfacing with other languages or systems that require character offsets, this will track the character offsets for you, accounting for any trimming that may have occurred.

## v0.24.0

### What's New

Update to pulldown-cmark 0.13.0 to improve Markdown parsing.

## v0.23.0

### What's New

Update to tree-sitter v0.25

## v0.22.0

### Breaking Changes

- Revert change to special token behavior in v0.21. This had many unintended side effects, and does not seem to be recommended for chunking.

## v0.21.0

### Breaking Changes

- Special tokens are now also encoded by both Huggingface and Tiktoken tokenizers. This is closer to the default behavior on the Python side, and should make sure if a model adds tokens at the beginning or end of a sequence, these are accounted for as well. This is especially important for embedding models that can add a special token to the beginning of the sequence, and the chunks generated didn't actually fit within the context window because of this.

### What's New

#### Rust

- MSRV is now 1.80 to remove dependency on once_cell.

## v0.20.1

### Fixes

- Python: correctly specify version for compatibility with `uv` installations.

## v0.20.0

### Breaking Changes

- Switched backing Unicode segmentation implementation from `unicode-segmentation` to `icu_segmenter`. This brings some modest performance gains, along with being able to leverage the official Unicode crate. There may be slight differences in chunk behavior in some edge cases, so treating this as a breaking change.

## v0.19.1

### What's New

- Python splitters have new `chunk_all` and `chunk_all_indices` method so the multiple texts can be processed in parallel. (For Rust, you should be able to use `rayon` to do this already)

## v0.19.0

### Breaking Changes

- Update to tokenizers v0.21

## v0.18.1

### What's New

- Ensure tokenizer sizers with truncation parameters count their overflow encodings

## v0.18.0

### Breaking

- Change supported `tiktoken-rs` version to `0.6.x`

## v0.17.1

### What's New

- Loosen `regex` crate version requirement

## v0.17.0

### Breaking Changes

- Support `tree-sitter@v0.24` for CodeSplitters.
- Due to a slight change in the backing unicode segmentation implementation, there are some slight shifts in behavior for CodeSplitters as well (in my tests, mostly that semicolons have a more logical grouping with previous content).

## v0.16.1

### What's New

Updates `pulldown-cmark` to `v0.12.1` to address an issue with high CPU usage for certain Markdown elements.

## v0.16.0

### Breaking Changes

- Update to `v0.23.0` of `tree-sitter`. There was a breaking change for language definitions, so this is also a breaking change for us, especially on the Python side, since we support passing the language in.
- Minimum Python version for the Python bindings is now 3.9 since 3.8 will be EOL next month.

#### Python

Make sure to upgrade to the latest version of your tree-sitter language package.

#### Rust

Make sure to upgrade to the latest version of your tree-sitter language package crate. These know have a `LANGUAGE` constant rather than a `language()` function.

```rust
// Before
tree_sitter_rust::language()
// After
tree_sitter_rust::LANGUAGE
```

### What's New

- `MarkdownSplitter` can better parse the Commonmark HS extension for Definition Lists.

## v0.15.0

### What's New

- Support version `0.20.0` of the `tokenizers` crate.

#### Python

- No longer cause a segmentation fault when using the wrong type for tree-sitter languages. Fixes [#265](https://github.com/benbrandt/text-splitter/issues/265)

## v0.14.1

### What's New

- Small performance improvements where checking the size of the chunk is avoided if we already know it is too small or we don't need to.
- Loosen dependency ranges for Rust crates to allow for more flexibility in the versions you can use.

## v0.14.0

### What's New

**Performance fixes for large documents.** The worst-case performance for certain documents was abysmal, leading to documents [that ran forever](https://github.com/benbrandt/text-splitter/issues/184). This release makes sure that in the worst case, the splitter won't be binary searching over the entire document, which it was before. This is prohibitively expensive especially for the tokenizer implementations, and now this should always have a safe upper bound to the search space.

For the "happy path", this new approach also led to big speed gains in the `CodeSplitter` (50%+ speed increase in some cases), marginal regressions in the `MarkdownSplitter`, and not much difference in the `TextSplitter`. But overall, the performance should be more consistent across documents, since it wasn't uncommon for a document with certain formatting to hit the worst-case scenario previously.

### Breaking Changes

- Chunk output may be slightly different because of the changes to the search optimizations. The previous optimization occasionally caused the splitter to stop too soon. For most cases, you may see no difference. It was most pronounced in the `MarkdownSplitter` at very small sizes, and any splitter using `RustTokenizers` because of its offset behavior.

#### Rust

- `ChunkSize` has been removed. This was a holdover from a previous internal optimization, which turned out to not be very accurate anyway.
- This makes implementing a custom `ChunkSizer` much easier, as you now only need to generate the size of the chunk as a `usize`. It often required in tokenization implementations to do more work to calculate the size as well, which is no longer necessary.

##### Before

```rust
pub trait ChunkSizer {
    // Required method
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize;
}
```

##### After

```rust
pub trait ChunkSizer {
    // Required method
    fn size(&self, chunk: &str) -> usize;
}
```

## v0.13.3

Fixes broken PyPI publish because of a bad dev dependency specification.

## v0.13.2

### What's New

New `CodeSplitter` for splitting code in any languages that [tree-sitter grammars](https://tree-sitter.github.io/tree-sitter/#parsers) are available for. It should provide decent chunks, but please provide feedback if you notice any strange behavior.

#### Rust Usage

```sh
cargo add text-splitter --features code
cargo add tree-sitter-<language>
```

```rust
use text_splitter::CodeSplitter;
// Default implementation uses character count for chunk size.
// Can also use all of the same tokenizer implementations as `TextSplitter`.
let splitter = CodeSplitter::new(tree_sitter_rust::language(), 1000).expect("Invalid tree-sitter language");

let chunks = splitter.chunks("your code file");
```

#### Python Usage

```python
from semantic_text_splitter import CodeSplitter
import tree_sitter_python

# Default implementation uses character count for chunk size.
# Can also use all of the same tokenizer implementations as `TextSplitter`.
splitter = CodeSplitter(tree_sitter_python.language(), capacity=1000)

chunks = splitter.chunks("your code file");
```

## v0.13.1

Fix a bug in the fallback logic to make sure we are still respecting the maximum bytes we should be searching in. Again, this only affects Markdown splitting at very small sizes.

## v0.13.0

### What's New / Breaking Changes

**Unicode Segmentation is now only used as a fallback**. This prioritizes the semantic levels of each splitter, and only uses Unicode grapheme/word/sentence segmentation when none of the semantic levels can be split at the desired capacity.

In most cases, this won't change the behavior of the splitter, and will likely mean that speed will improve because it is able to skip several semantic levels at the start, acting as a bisect or binary search, and only go back to the lower levels if it can't fit.

However, for the `MarkdownSplitter` at very small sizes (i.e., less than 16 tokens), this may produce different output, becuase prior to this change, the splitter may have used Unicode sentence segmentation instead of the Markdown semantic levels, due to an optimization in the level selection. Now, the splitter will prioritize the parsed Markdown levels before it falls back to Unicode segmentation, which preserves better structure at small sizes.

**So, it is likely in most cases, this is a non-breaking update**. However, if you were using extremely small chunk sizes for Markdown, the behavior is different, and I wanted to inidicate that with a major version bump.

## v0.12.3

### Bug Fix

Remove leftover `dbg!` statements in chunk overlap code [#154](https://github.com/benbrandt/text-splitter/pull/164) 🤦🏻‍♂️

Apologies if I spammed your logs!

## v0.12.2

### What's New

**Support for chunk overlapping:** Several of you have been waiting on this for awhile now, and I am happy to say that chunk overlapping is now available in a way that still stays true to the spirit of finding good semantic break points.

When a new chunk is emitted, if chunk overlapping is enabled, the splitter will look back at the semantic sections of the current level and pull in as many as possible that fit within the overlap window. **This does mean that none can be taken**, which is often the case when close to a higher semantic level boundary.

When it will almost always produce an overlap is when the current semantic level couldn't be fit into a single chunk, and it provides overlapping sections since we may not have found a good break point in the middle of the section. Which seems to be the main motivation for using chunk overlapping in the first place.

#### Rust Usage

```rust
let chunk_config = ChunkConfig::new(256)
    // .with_sizer(sizer) // Optional tokenizer or other chunk sizer impl
    .with_overlap(64)
    .expect("Overlap must be less than desired chunk capacity");
let splitter = TextSplitter::new(chunk_config); // Or MarkdownSplitter
```

#### Python Usage

```python
splitter = TextSplitter(256, overlap=64) # or any of the class methods to use a tokenizer
```

## v0.12.1

### What's New

- [`rust_tokenizers`](https://crates.io/crates/rust_tokenizers) support has been added to the Rust crate.

## v0.12.0

### What's New

This release is a big API change to pull all chunk configuration options into the same place, at initialization of the splitters. This was motivated by two things:

1. These settings are all important to deciding how to split the text for a given use case, and in practice I saw them often being set together anyway.
2. To prep the library for new features like chunk overlap, where error handling has to be introduced to make sure that invariants are kept between all of the settings. These errors should be handled as sson as possible before chunking the text.

Overall, I think this has aligned the library with the usage I have seen in the wild, and pulls all of the settings for the "domain" of chunking into a single unit.

### Breaking Changes

#### Rust

- **Trimming is now enabled by default**. This brings the Rust crate in alignment with the Python package. But for every use case I saw, this was already being set to `true`, and this does logically make sense as the default behavior.
- `TextSplitter` and `MarkdownSplitter` now take a `ChunkConfig` in their `::new` method
  - This bring the `ChunkSizer`, `ChunkCapacity` and `trim` settings into a single struct that can be instantiated with a builder-lite pattern.
  - `with_trim_chunks` method has been removed from `TextSplitter` and `MarkdownSplitter`. You can now set `trim` in the `ChunkConfig` struct.
- `ChunkCapacity` is now a struct instead of a Trait. If you were using a custom `ChunkCapacity`, you can change your `impl` to a `From<TYPE> for ChunkCapacity` instead. and you should be able to still pass it in to all of the same methods.
  - This also means `ChunkSizer`s take a concrete type in their method instead of an impl

##### Migration Examples

**Default settings:**

```rust
/// Before
let splitter = TextSplitter::default().with_trim_chunks(true);
let chunks = splitter.chunks("your document text", 500);

/// After
let splitter = TextSplitter::new(500);
let chunks = splitter.chunks("your document text");
```

**Hugging Face Tokenizers:**

```rust
/// Before
let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
let splitter = TextSplitter::new(tokenizer).with_trim_chunks(true);
let chunks = splitter.chunks("your document text", 500);

/// After
let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
let splitter = TextSplitter::new(ChunkConfig::new(500).with_sizer(tokenizer));
let chunks = splitter.chunks("your document text");
```

**Tiktoken:**

```rust
/// Before
let tokenizer = cl100k_base().unwrap();
let splitter = TextSplitter::new(tokenizer).with_trim_chunks(true);
let chunks = splitter.chunks("your document text", 500);

/// After
let tokenizer = cl100k_base().unwrap();
let splitter = TextSplitter::new(ChunkConfig::new(500).with_sizer(tokenizer));
let chunks = splitter.chunks("your document text");
```

**Ranges:**

```rust
/// Before
let splitter = TextSplitter::default().with_trim_chunks(true);
let chunks = splitter.chunks("your document text", 500..2000);

/// After
let splitter = TextSplitter::new(500..2000);
let chunks = splitter.chunks("your document text");
```

**Markdown:**

```rust
/// Before
let splitter = MarkdownSplitter::default().with_trim_chunks(true);
let chunks = splitter.chunks("your document text", 500);

/// After
let splitter = MarkdownSplitter::new(500);
let chunks = splitter.chunks("your document text");
```

**ChunkSizer impls**

```rust
pub trait ChunkSizer {
    /// Before
    fn chunk_size(&self, chunk: &str, capacity: &impl ChunkCapacity) -> ChunkSize;
    /// After
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize;
}
```

**ChunkCapacity impls**

```rust
/// Before
impl ChunkCapacity for Range<usize> {
    fn start(&self) -> Option<usize> {
        Some(self.start)
    }

    fn end(&self) -> usize {
        self.end.saturating_sub(1).max(self.start)
    }
}

/// After
impl From<Range<usize>> for ChunkCapacity {
    fn from(range: Range<usize>) -> Self {
        ChunkCapacity::new(range.start)
            .with_max(range.end.saturating_sub(1).max(range.start))
            .expect("invalid range")
    }
}
```

#### Python

- Chunk `capacity` is now a required arguement in the `__init__` and classmethods of `TextSplitter` and `MarkdownSplitter`
- `trim_chunks` parameter is now just `trim` in the `__init__` and classmethods of `TextSplitter` and `MarkdownSplitter`

##### Migration Examples

**Default settings:**

```python
# Before
splitter = TextSplitter()
chunks = splitter.chunks("your document text", 500)

# After
splitter = TextSplitter(500)
chunks = splitter.chunks("your document text")
```

**Ranges:**

```python
# Before
splitter = TextSplitter()
chunks = splitter.chunks("your document text", (200,1000))

# After
splitter = TextSplitter((200,1000))
chunks = splitter.chunks("your document text")
```

**Hugging Face Tokenizers:**

```python
# Before
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
splitter = TextSplitter.from_huggingface_tokenizer(tokenizer)
chunks = splitter.chunks("your document text", 500)

# After
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, 500)
chunks = splitter.chunks("your document text")
```

**Tiktoken:**

```python
# Before
splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo")
chunks = splitter.chunks("your document text", 500)

# After
splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", 500)
chunks = splitter.chunks("your document text")
```

**Custom callback:**

```python
# Before
splitter = TextSplitter.from_callback(lambda text: len(text))
chunks = splitter.chunks("your document text", 500)

# After
splitter = TextSplitter.from_callback(lambda text: len(text), 500)
chunks = splitter.chunks("your document text")
```

**Markdown:**

```python
# Before
splitter = MarkdownSplitter()
chunks = splitter.chunks("your document text", 500)

# After
splitter = MarkdownSplitter(500)
chunks = splitter.chunks("your document text")
```

## v0.11.0

### Breaking Changes

- Bump tokenizers from 0.15.2 to 0.19.1

### Other updates

- Bump either from 1.10.0 to 1.11.0
- Bump pyo3 from 0.21.1 to 0.21.2

## v0.10.0

### Breaking Changes

**Improved (but different) Markdown split points** [#137](https://github.com/benbrandt/text-splitter/pull/137). In hindsight, the levels used for determining split points in Markdown text were too granular, which led to some strange split points.
Many more element types were consolidated into the same levels, which should still provide a good balance between splitting at the right points and not splitting too often.

Because the output of the `MarkdownSplitter` will be substantially different, especially for smaller chunk sizes, this is considered a breaking change.

## v0.9.1

### What's New

Python `TextSplitter` and `MarkdownSplitter` now both provide a new `chunk_indices` method that returns a list not only of chunks, but also their corresponding character offsets relative to the original text. This should allow for different string comparison and matching operations on the chunks.

```python
def chunk_indices(
    self, text: str, chunk_capacity: Union[int, Tuple[int, int]]
) -> List[Tuple[int, str]]:
    ...
```

A similar method already existed on the Rust side. The key difference is that these offsets are **character** not **byte** offsets. For Rust strings, it is usually helpful to have the byte offset, but in Python, most string methods and operations deal with character indices.

## v0.9.0

### What's New

[More robust handling of Hugging Face tokenizers as chunk sizers.](https://github.com/benbrandt/text-splitter/pull/131)

- **Tokenizers with padding enabled no longer count padding tokens when generating chunks**. This caused some unexpected behavior, especially if the chunk capacity didn't perfectly line up with the padding size(s). Now, the tokenizer's padding token is ignored when counting the number of tokens generated in a chunk.
- In the process, it also became clear there were some false assumptions about how the byte offset ranges were calculated for each token. This has been fixed, and the byte offset ranges should now be more accurate when determining the boundaries of each token. This only affects some optimizations in chunk sizing, and should not affect the actual chunk output.

### Breaking Changes

There should only be breaking chunk output for those of you using a Hugging Face tokenizer with padding enabled. Because padding tokens are no longer counted, the chunks will likely be larger than before, and closer to the desired behavior.

**Note:** This will mean the generated chunks may also be larger than the chunk capacity when tokenized, because padding tokens will be added when you tokenize the chunk. The chunk capacity for these tokenizers reflects the number of tokens used in the text, not necessarily the number of tokens that the tokenizer will generate in total.

## v0.8.1

### What's New

- Updates to documentation and examples.
- Update pyo3 to 0.21.0 in Python package, which should bring some performance improvements.

## v0.8.0

### What's New

[Significantly fewer allocations](https://github.com/benbrandt/text-splitter/pull/121) necessary when generating chunks. This should result in a performance improvement for most use cases. This was achieved by both reusing pre-allocated collections, as well as memoizing chunk size calculations since that is often the bottleneck, and tokenizer libraries tend to be very allocation heavy!

Benchmarks show:

- **20-40% fewer** allocations caused by the core algorithm.
- **Up to 20% fewer** allocations when using tokenizers to calculate chunk sizes.
- In some cases, especially with Markdown, these improvements can also result in **up to 20% faster** chunk generation.

### Breaking Changes

- There was a bug in the `MarkdownSplitter` logic that caused some strange split points.
- The `Text` semantic level in `MarkdownSplitter` has been merged with inline elements to also find better split points inside content.
- Fixed a bug that could cause the algorithm to use a lower semantic level than necessary on occasion. This mostly impacted the `MarkdownSplitter`, but there were same cases of different behavior in the `TextSplitter` as well if chunks are not trimmed.

All of the above can cause different chunks to be output than before, depending on the text. So, even though these are bug fixes to bring intended behavior, they are being treated as a major version bump.

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

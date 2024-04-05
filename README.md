# text-splitter

[![Docs](https://docs.rs/text-splitter/badge.svg)](https://docs.rs/text-splitter/)
[![Licence](https://img.shields.io/crates/l/text-splitter)](https://github.com/benbrandt/text-splitter/blob/main/LICENSE.txt)
[![Crates.io](https://img.shields.io/crates/v/text-splitter)](https://crates.io/crates/text-splitter)
[![codecov](https://codecov.io/github/benbrandt/text-splitter/branch/main/graph/badge.svg?token=TUF1IAI7G7)](https://codecov.io/github/benbrandt/text-splitter)

- **Rust Crate**: [text-splitter](https://crates.io/crates/text-splitter)
- **Python Bindings**: [semantic-text-splitter](https://pypi.org/project/semantic-text-splitter/) (unfortunately couldn't acquire the same package name)

Large language models (LLMs) can be used for many tasks, but often have a limited context size that can be smaller than documents you might want to use. To use documents of larger length, you often have to split your text into chunks to fit within this context size.

This crate provides methods for splitting longer pieces of text into smaller chunks, aiming to maximize a desired chunk size, but still splitting at semantically sensible boundaries whenever possible.

## Get Started

Add it to your project with

```sh
cargo add text-splitter
```

### By Number of Characters

The simplest way to use this crate is to use the default implementation, which uses character count for chunk size.

```rust
use text_splitter::{Characters, TextSplitter};

// Maximum number of characters in a chunk
let max_characters = 1000;
// Default implementation uses character count for chunk size
let splitter = TextSplitter::default()
    // Optionally can also have the splitter trim whitespace for you
    .with_trim_chunks(true);

let chunks = splitter.chunks("your document text", max_characters);
println!("{}", chunks.count())
```

### With Hugging Face Tokenizer

Requires the `tokenizers` feature to be activated and adding `tokenizers` to dependencies. The example below, using `from_pretrained()`, also requires tokenizers `http` feature to be enabled.

```sh
cargo add text-splitter --features tokenizers
cargo add tokenizers --features http
```

```rust
use text_splitter::TextSplitter;
// Can also use anything else that implements the ChunkSizer
// trait from the text_splitter crate.
use tokenizers::Tokenizer;

let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
let max_tokens = 1000;
let splitter = TextSplitter::new(tokenizer)
    // Optionally can also have the splitter trim whitespace for you
    .with_trim_chunks(true);

let chunks = splitter.chunks("your document text", max_tokens);
println!("{}", chunks.count())
```

### With Tiktoken Tokenizer

Requires the `tiktoken-rs` feature to be activated and adding `tiktoken-rs` to dependencies.

```sh
cargo add text-splitter --features tiktoken-rs
cargo add tiktoken-rs
```

```rust
use text_splitter::TextSplitter;
// Can also use anything else that implements the ChunkSizer
// trait from the text_splitter crate.
use tiktoken_rs::cl100k_base;

let tokenizer = cl100k_base().unwrap();
let max_tokens = 1000;
let splitter = TextSplitter::new(tokenizer)
    // Optionally can also have the splitter trim whitespace for you
    .with_trim_chunks(true);

let chunks = splitter.chunks("your document text", max_tokens);
println!("{}", chunks.count())
```

### Using a Range for Chunk Capacity

You also have the option of specifying your chunk capacity as a range.

Once a chunk has reached a length that falls within the range it will be returned.

It is always possible that a chunk may be returned that is less than the `start` value, as adding the next piece of text may have made it larger than the `end` capacity.

```rust
use text_splitter::{Characters, TextSplitter};

// Maximum number of characters in a chunk. Will fill up the
// chunk until it is somewhere in this range.
let max_characters = 500..2000;
// Default implementation uses character count for chunk size
let splitter = TextSplitter::default().with_trim_chunks(true);

let chunks = splitter.chunks("your document text", max_characters);
println!("{}", chunks.count())
```

### Markdown

All of the above examples also can also work with Markdown text. If you enable the `markdown` feature, you can use the `MarkdownSplitter` in the same ways as the `TextSplitter`.

```sh
cargo add text-splitter --features markdown
```

```rust
use text_splitter::MarkdownSplitter;
// Maximum number of characters in a chunk. Can also use a range.
let max_characters = 1000;
// Default implementation uses character count for chunk size.
// Can also use all of the same tokenizer implementations as `TextSplitter`.
let splitter = MarkdownSplitter::default()
    // Optionally can also have the splitter trim whitespace for you
    .with_trim_chunks(true);

let chunks = splitter.chunks("# Header\n\nyour document text", max_characters);
println!("{}", chunks.count())
```

## Method

To preserve as much semantic meaning within a chunk as possible, each chunk is composed of the largest semantic units that can fit in the next given chunk. For each splitter type, there is a defined set of semantic levels. Here is an example of the steps used:

1. Split the text by a increasing semantic levels.
2. Check the first item for each level and select the highest level whose first item still fits within the chunk size.
3. Merge as many of these neighboring sections of this level or above into a chunk to maximize chunk length. Boundaries of higher semantic levels are always included when merging, so that the chunk doesn't inadvertantly cross semantic boundaries.

The boundaries used to split the text if using the `chunks` method, in ascending order:

### `TextSplitter` Semantic Levels

1. Characters
2. [Unicode Grapheme Cluster Boundaries](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
3. [Unicode Word Boundaries](https://www.unicode.org/reports/tr29/#Word_Boundaries)
4. [Unicode Sentence Boundaries](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
5. Ascending sequence length of newlines. (Newline is `\r\n`, `\n`, or `\r`) Each unique length of consecutive newline sequences is treated as its own semantic level. So a sequence of 2 newlines is a higher level than a sequence of 1 newline, and so on.

Splitting doesn't occur below the character level, otherwise you could get partial bytes of a char, which may not be a valid unicode str.

### `MarkdownSplitter` Semantic Levels

Markdown is parsed according to the CommonMark spec, along with some optional features such as GitHub Flavored Markdown.

1. Characters
2. [Unicode Grapheme Cluster Boundaries](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
3. [Unicode Word Boundaries](https://www.unicode.org/reports/tr29/#Word_Boundaries)
4. [Unicode Sentence Boundaries](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
5. Soft line breaks (single newline) which isn't necessarily a new element in Markdown.
6. Inline elements such as: text nodes, emphasis, strong, strikethrough, link, image, table cells, inline code, footnote references, task list markers, and inline html.
7. Block elements suce as: paragraphs, code blocks, footnote definitions, metadata. Also, a block quote or row/item within a table or list that can contain other "block" type elements, and a list or table that contains items.
8. Thematic breaks or horizontal rules.
9. Headings by level

Splitting doesn't occur below the character level, otherwise you could get partial bytes of a char, which may not be a valid unicode str.

### Note on sentences

There are lots of methods of determining sentence breaks, all to varying degrees of accuracy, and many requiring ML models to do so. Rather than trying to find the perfect sentence breaks, we rely on unicode method of sentence boundaries, which in most cases is good enough for finding a decent semantic breaking point if a paragraph is too large, and avoids the performance penalties of many other methods.

## Feature Flags

### Document Format Support

| Feature    | Description                                                                                   |
| ---------- | --------------------------------------------------------------------------------------------- |
| `markdown` | Enables the `MarkdownSplitter` struct for parsing Markdown documents via the CommonMark spec. |

### Tokenizer Support

| Dependency Feature | Version Supported | Description                                                                                                                                                                    |
| ------------------ | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tiktoken-rs`      | `0.5.8`           | Enables `(Text/Markdown)Splitter::new` to take `tiktoken_rs::CoreBPE` as an argument. This is useful for splitting text for OpenAI models.                                     |
| `tokenizers`       | `0.15.2`          | Enables `(Text/Markdown)Splitter::new` to take `tokenizers::Tokenizer` as an argument. This is useful for splitting text models that have a Hugging Face-compatible tokenizer. |

## Inspiration

This crate was inspired by [LangChain's TextSplitter](https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html#langchain_text_splitters.character.RecursiveCharacterTextSplitter). But, looking into the implementation, there was potential for better performance as well as better semantic chunking.

A big thank you to the unicode-rs team for their [unicode-segmentation](https://crates.io/crates/unicode-segmentation) crate that manages a lot of the complexity of matching the Unicode rules for words and sentences.

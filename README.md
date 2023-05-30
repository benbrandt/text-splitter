# text-splitter

[![Docs](https://docs.rs/text-splitter/badge.svg)](https://docs.rs/text-splitter/)
[![Licence](https://img.shields.io/crates/l/text-splitter)](https://github.com/benbrandt/text-splitter/blob/main/LICENSE.txt)
[![Crates.io](https://img.shields.io/crates/v/text-splitter)](https://crates.io/crates/text-splitter)
[![codecov](https://codecov.io/github/benbrandt/text-splitter/branch/main/graph/badge.svg?token=TUF1IAI7G7)](https://codecov.io/github/benbrandt/text-splitter)

Large language models (LLMs) can be used for many tasks, but often have a limited context size that can be smaller than documents you might want to use. To use documents of larger length, you often have to split your text into chunks to fit within this context size.

This crate provides methods for splitting longer pieces of text into smaller chunks, aiming to maximize a desired chunk size, but still splitting at semantically sensible boundaries whenever possible.

## Get Started

### By Number of Characters

```rust
use text_splitter::{Characters, TextSplitter};

// Maximum number of characters in a chunk
let max_characters = 1000;
// Default implementation uses character count for chunk size
let splitter = TextSplitter::default()
    // Optionally can also have the splitter trim whitespace for you
    .with_trim_chunks(true);

let chunks = splitter.chunks("your document text", max_characters);
```

### By Tokens

```rust
use text_splitter::TextSplitter;
// Can also use tiktoken-rs, or anything that implements the ChunkSize
// trait from the text_splitter crate.
use tokenizers::Tokenizer;

let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
let max_tokens = 1000;
let splitter = TextSplitter::new(tokenizer)
    // Optionally can also have the splitter trim whitespace for you
    .with_trim_chunks(true);

let chunks = splitter.chunks("your document text", max_tokens);
```

## Method

To preserve as much semantic meaning within a chunk as possible, a recursive approach is used, starting at larger semantic units and, if that is too large, breaking it up into the next largest unit. Here is an example of the steps used:

1. Split the text by a given level
2. For each section, does it fit within the chunk size?
   - Yes. Merge as many of these neighboring sections into a chunk as possible to maximize chunk length.
   - No. Split by the next level and repeat.

The boundaries used to split the text if using the top-level `chunks` method, in descending length:

1. Descending sequence length of newlines. (Newline is `\r\n`, `\n`, or `\r`) Each unique length of consecutive newline sequences is treated as its own semantic level.
2. [Unicode Sentence Boundaries](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
3. [Unicode Word Boundaries](https://www.unicode.org/reports/tr29/#Word_Boundaries)
4. [Unicode Grapheme Cluster Boundaries](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
5. Characters

Splitting doesn't occur below the character level, otherwise you could get partial bytes of a char, which may not be a valid unicode str.

_Note on sentences:_ There are lots of methods of determining sentence breaks, all to varying degrees of accuracy, and many requiring ML models to do so. Rather than trying to find the perfect sentence breaks, we rely on unicode method of sentence boundaries, which in most cases is good enough for finding a decent semantic breaking point if a paragraph is too large, and avoids the performance penalties of many other methods.

## Inspiration

This crate was inspired by [LangChain's TextSplitter](https://python.langchain.com/en/latest/modules/indexes/text_splitters/examples/recursive_text_splitter.html). But, looking into the implementation, there was potential for better performance as well as better semantic chunking.

A big thank you to the unicode-rs team for their [unicode-segmentation](https://crates.io/crates/unicode-segmentation) crate that manages a lot of the complexity of matching the Unicode rules for words and sentences.

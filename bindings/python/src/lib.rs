//! Python Bindings for text-splitter crate

#![warn(
    clippy::pedantic,
    future_incompatible,
    missing_debug_implementations,
    missing_docs,
    nonstandard_style,
    rust_2018_compatibility,
    rust_2018_idioms,
    rust_2021_compatibility,
    unused
)]
// pyo3 uses this
#![allow(elided_lifetimes_in_paths)]

use ::text_splitter::{Characters, ChunkCapacity, TextSplitter};
use pyo3::prelude::*;

/// Custom chunk capacity for python to make it easier to work
/// with python arguments
#[derive(Debug, FromPyObject)]
enum PyChunkCapacity {
    #[pyo3(transparent, annotation = "int")]
    Int(usize),
    #[pyo3(annotation = "tuple[int, int]")]
    IntTuple(usize, usize),
}

impl ChunkCapacity for PyChunkCapacity {
    fn start(&self) -> Option<usize> {
        match self {
            Self::Int(_) => None,
            Self::IntTuple(start, _) => Some(*start),
        }
    }

    fn end(&self) -> usize {
        match self {
            Self::Int(end) | Self::IntTuple(_, end) => *end,
        }
    }
}

/**
Plain-text splitter. Recursively splits chunks into the largest semantic units that fit within the chunk size. Also will attempt to merge neighboring chunks if they can fit within the given chunk size.

### By Number of Characters

```python
from text_splitter import CharacterTextSplitter

# Maximum number of characters in a chunk
max_characters = 1000
# Optionally can also have the splitter trim whitespace for you
splitter = CharacterTextSplitter(trim_chunks=True)

chunks = splitter.chunks("your document text", max_characters)
```

### Using a Range for Chunk Capacity

You also have the option of specifying your chunk capacity as a range.

Once a chunk has reached a length that falls within the range it will be returned.

It is always possible that a chunk may be returned that is less than the `start` value, as adding the next piece of text may have made it larger than the `end` capacity.

```python
from text_splitter import CharacterTextSplitter

# Optionally can also have the splitter trim whitespace for you
splitter = CharacterTextSplitter()

# Maximum number of characters in a chunk. Will fill up the
# chunk until it is somewhere in this range.
chunks = splitter.chunks("your document text", chunk_capacity=(200,1000))
```
**/
#[pyclass]
struct CharacterTextSplitter {
    splitter: TextSplitter<Characters>,
}

#[pymethods]
impl CharacterTextSplitter {
    /// Specify whether chunks should have whitespace trimmed from the
    /// beginning and end or not.
    ///
    /// If `False` (default), joining all chunks should return the original
    /// string.
    /// If `True`, all chunks will have whitespace removed from beginning and end.
    #[new]
    #[pyo3(signature = (trim_chunks=false))]
    fn new(trim_chunks: bool) -> Self {
        Self {
            splitter: TextSplitter::default().with_trim_chunks(trim_chunks),
        }
    }

    /// Generate a list of chunks from a given text. Each chunk will be up to the `chunk_capacity`.
    ///
    /// ## Method
    ///
    /// To preserve as much semantic meaning within a chunk as possible, a recursive approach is used, starting at larger semantic units and, if that is too large, breaking it up into the next largest unit. Here is an example of the steps used:
    ///
    /// 1. Split the text by a given level
    /// 2. For each section, does it fit within the chunk size?
    ///   a. Yes. Merge as many of these neighboring sections into a chunk as possible to maximize chunk length.
    ///   b. No. Split by the next level and repeat.
    ///
    /// The boundaries used to split the text if using the top-level `split` method, in descending length:
    ///
    /// 1. Descending sequence length of newlines. (Newline is `\r\n`, `\n`, or `\r`) Each unique length of consecutive newline sequences is treated as its own semantic level.
    /// 2. [Unicode Sentence Boundaries](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
    /// 3. [Unicode Word Boundaries](https://www.unicode.org/reports/tr29/#Word_Boundaries)
    /// 4. [Unicode Grapheme Cluster Boundaries](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
    /// 5. Characters
    ///
    /// Splitting doesn't occur below the character level, otherwise you could get partial
    /// bytes of a char, which may not be a valid unicode str.
    fn chunks<'text, 'splitter: 'text>(
        &'splitter self,
        text: &'text str,
        chunk_capacity: PyChunkCapacity,
    ) -> Vec<&'text str> {
        self.splitter.chunks(text, chunk_capacity).collect()
    }
}

/**
# text-splitter

[![Licence](https://img.shields.io/crates/l/text-splitter)](https://github.com/benbrandt/text-splitter/blob/main/LICENSE.txt)

Large language models (LLMs) can be used for many tasks, but often have a limited context size that can be smaller than documents you might want to use. To use documents of larger length, you often have to split your text into chunks to fit within this context size.

This crate provides methods for splitting longer pieces of text into smaller chunks, aiming to maximize a desired chunk size, but still splitting at semantically sensible boundaries whenever possible.

## Get Started

### By Number of Characters

```python
from text_splitter import CharacterTextSplitter

# Maximum number of characters in a chunk
max_characters = 1000
# Optionally can also have the splitter trim whitespace for you
splitter = CharacterTextSplitter(trim_chunks=True)

chunks = splitter.chunks("your document text", max_characters)
```

### Using a Range for Chunk Capacity

You also have the option of specifying your chunk capacity as a range.

Once a chunk has reached a length that falls within the range it will be returned.

It is always possible that a chunk may be returned that is less than the `start` value, as adding the next piece of text may have made it larger than the `end` capacity.

```python
from text_splitter import CharacterTextSplitter

# Optionally can also have the splitter trim whitespace for you
splitter = CharacterTextSplitter()

# Maximum number of characters in a chunk. Will fill up the
# chunk until it is somewhere in this range.
chunks = splitter.chunks(
    "your document text",
    chunk_capacity_start=200,
    chunk_capacity_end=1000
)
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
**/
#[pymodule]
fn text_splitter(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CharacterTextSplitter>()?;
    Ok(())
}

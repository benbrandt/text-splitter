//! Text Splitter
//!
//! Large language models (LLMs) have lots of amazing use cases. But often they
//! have a limited context size that is smaller than larger documents. In order
//! to use documents of larger length, you often have to split your text into
//! chunks to fit within this context size.
//!
//! This crate provides methods for doing so by trying to maximize a desired
//! chunk size, but still splitting at semantic units whenever possible.

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

use itertools::Itertools;

/// Default plain-text splitter. Recursively splits chunks into the smallest
/// semantic units that fit within the chunk size. Also will attempt to merge
/// neighboring chunks if they can fit within the given chunk size.
#[derive(Clone, Copy, Debug)]
pub struct TextSplitter {
    /// Maximum size of a chunk (measured in characters)
    max_chunk_size: usize,
}

impl TextSplitter {
    /// Creates a new [`TextSplitter`].
    ///
    /// ```
    /// use text_splitter::TextSplitter;
    ///
    /// let splitter = TextSplitter::new(100);
    /// ```
    #[must_use]
    pub fn new(max_chunk_size: usize) -> Self {
        Self { max_chunk_size }
    }

    /// Generate a list of chunks from a given text. Each chunk will be up to
    /// the `max_chunk_size`.
    ///
    /// ```
    /// use text_splitter::TextSplitter;
    ///
    /// let splitter = TextSplitter::new(100);
    /// let text = "Some text from a document";
    /// let chunks = splitter.chunks(text);
    /// ```
    pub fn chunks<'a, 'b: 'a>(&'a self, text: &'b str) -> impl Iterator<Item = &'b str> + 'a {
        // Lowest-level split, since we are dealing with chars by default
        self.split_chars(text)
    }

    /// Split a given text by chars where each chunk is within the max chunk
    /// size.
    fn split_chars<'a, 'b: 'a>(&'a self, text: &'b str) -> impl Iterator<Item = &'b str> + 'a {
        text.char_indices().batching(|it| {
            let (start, end) = it
                .take(self.max_chunk_size)
                .fold::<(Option<usize>, usize), _>((None, 0), |(start, _), (i, c)| {
                    (start.or(Some(i)), i + c.len_utf8())
                });
            start.map(|start| text.get(start..end).unwrap())
        })
    }
}

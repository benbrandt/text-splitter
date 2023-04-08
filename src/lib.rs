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
pub struct TextSplitter {
    /// Maximum size of a chunk (measured by length_fn)
    max_chunk_size: usize,
    /// Method of calculating chunk length. By default done at the character level.
    length_fn: Box<dyn Fn(&str) -> usize>,
}

impl std::fmt::Debug for TextSplitter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextSplitter")
            .field("max_chunk_size", &self.max_chunk_size)
            .finish()
    }
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
        Self {
            max_chunk_size,
            length_fn: Box::new(|text| text.chars().count()),
        }
    }

    /// Specify a custom function for calculating the length of a chunk. For
    /// example, using chars instead of bytes.
    ///
    /// ```
    /// use text_splitter::TextSplitter;
    ///
    /// let splitter = TextSplitter::new(100).with_length_fn(|text| text.chars().count());
    /// ```
    #[must_use]
    pub fn with_length_fn(mut self, length_fn: impl Fn(&str) -> usize + 'static) -> Self {
        self.length_fn = Box::new(length_fn);
        self
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
        // Lowest-level split so we always have valid unicode chars
        self.split_chars(text)
    }

    /// Is the given text within the chunk size?
    fn is_within_chunk_size(&self, text: &str) -> bool {
        (self.length_fn)(text) <= self.max_chunk_size
    }

    /// Split a given text by chars where each chunk is within the max chunk
    /// size.
    fn split_chars<'a, 'b: 'a>(&'a self, text: &'b str) -> impl Iterator<Item = &'b str> + 'a {
        text.char_indices().peekable().batching(move |it| {
            let mut peek_start = None;
            let (start, end) = it
                .peeking_take_while(move |(i, c)| {
                    if peek_start.is_none() {
                        peek_start = Some(*i);
                    }
                    let Some(text) = text.get(peek_start.unwrap_or(*i)..*i + c.len_utf8()) else {
                        // Continue on, not a valid split
                        return true;
                    };
                    if self.is_within_chunk_size(text) {
                        true
                    } else {
                        peek_start = None;
                        false
                    }
                })
                .fold::<(Option<usize>, usize), _>((None, 0), |(start, _), (i, c)| {
                    (start.or(Some(i)), i + c.len_utf8())
                });
            start.and_then(|start| text.get(start..end))
        })
    }
}

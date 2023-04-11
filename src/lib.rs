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

use core::{fmt, iter::once};

use either::Either;
use itertools::Itertools;
use unicode_segmentation::UnicodeSegmentation;

/// Default plain-text splitter. Recursively splits chunks into the smallest
/// semantic units that fit within the chunk size. Also will attempt to merge
/// neighboring chunks if they can fit within the given chunk size.
pub struct TextSplitter {
    /// Maximum size of a chunk (measured by length_fn)
    max_chunk_size: usize,
    /// Method of calculating chunk length. By default done at the character level.
    length_fn: Box<dyn Fn(&str) -> usize>,
}

impl fmt::Debug for TextSplitter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

    /// Is the given text within the chunk size?
    fn is_within_chunk_size(&self, chunk: &str) -> bool {
        (self.length_fn)(chunk) <= self.max_chunk_size
    }

    /// Generate a list of chunks from a given text. Each chunk will be up to
    /// the `max_chunk_size`.
    ///
    /// ```
    /// use text_splitter::TextSplitter;
    ///
    /// let splitter = TextSplitter::new(100);
    /// let text = "Some text from a document";
    /// let chunks = splitter.chunk_by_graphemes(text);
    /// ```
    pub fn chunk_by_graphemes<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = &'b str> + 'a {
        self.chunk_by_grapheme_indices(text).map(|(_, t)| t)
    }

    /// Preserve Unicode graphemes where possible. Char iter would break them
    /// up by default.
    fn chunk_by_grapheme_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = (usize, &'b str)> + 'a {
        text.grapheme_indices(true)
            .flat_map(|(i, grapheme)| {
                // If grapheme is too large, do char chunking
                if self.is_within_chunk_size(grapheme) {
                    Either::Left(once((i, grapheme)))
                } else {
                    Either::Right(
                        self.chunk_by_char_indices(grapheme)
                            .map(move |(ci, c)| (ci + i, c)),
                    )
                }
            })
            .peekable()
            .batching(move |it| {
                // Otherwise keep grabbing more graphemes
                let mut peek_start = None;
                let (start, end) = it
                    .peeking_take_while(move |(i, g)| {
                        let chunk = text
                            .get(*peek_start.get_or_insert(*i)..*i + g.len())
                            .expect("grapheme should be valid");
                        if self.is_within_chunk_size(chunk) {
                            true
                        } else {
                            peek_start = None;
                            false
                        }
                    })
                    .fold::<(Option<usize>, usize), _>((None, 0), |(start, _), (i, g)| {
                        (start.or(Some(i)), i + g.len())
                    });
                start.and_then(|start| text.get(start..end).map(|t| (start, t)))
            })
    }

    /// Generate a list of chunks from a given text. Each chunk will be up to
    /// the `max_chunk_size`.
    ///
    /// If a text is too large, each chunk will fit as many `char`s as
    /// possible.
    ///
    /// If you chunk size is smaller than a given character, it will get
    /// filtered out, otherwise you would get just partial bytes of a char
    /// that might not be a valid unicode str.
    ///
    /// ```
    /// use text_splitter::TextSplitter;
    ///
    /// let splitter = TextSplitter::new(100);
    /// let text = "Some text from a document";
    /// let chunks = splitter.chunk_by_chars(text);
    /// ```
    pub fn chunk_by_chars<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = &'b str> + 'a {
        self.chunk_by_char_indices(text).map(|(_, t)| t)
    }

    /// Split a given text by chars where each chunk is within the max chunk
    /// size.
    fn chunk_by_char_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = (usize, &'b str)> + 'a {
        text.char_indices().peekable().batching(move |it| {
            let mut peek_start = None;
            let (start, end) = it
                .peeking_take_while(move |(i, c)| {
                    let chunk = text
                        .get(*peek_start.get_or_insert(*i)..*i + c.len_utf8())
                        .expect("char should be valid");
                    if self.is_within_chunk_size(chunk) {
                        true
                    } else {
                        peek_start = None;
                        false
                    }
                })
                .fold::<(Option<usize>, usize), _>((None, 0), |(start, _), (i, c)| {
                    (start.or(Some(i)), i + c.len_utf8())
                });
            start.and_then(|start| text.get(start..end).map(|t| (start, t)))
        })
    }
}

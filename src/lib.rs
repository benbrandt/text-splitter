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
use once_cell::sync::Lazy;
use regex::Regex;
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

// Lazy's so that we don't have to compile them more than once
/// Any sequence of 2 or more newlines
static DOUBLE_NEWLINE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\r\n){2,}|\r{2,}|\n{2,}").unwrap());
/// Fallback for anything else
static NEWLINE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\r\n)+|\r+|\n+").unwrap());

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

    /// Internal method to handle chunk splitting for anything above char level
    fn generate_chunks_from_str_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
        it: impl Iterator<Item = (usize, &'b str)> + 'a,
    ) -> impl Iterator<Item = (usize, &'b str)> + 'a {
        it.peekable()
            .batching(move |it| {
                // Otherwise keep grabbing more graphemes
                let mut peek_start = None;
                let (start, end) = it
                    .peeking_take_while(move |(i, str)| {
                        let chunk = text
                            .get(*peek_start.get_or_insert(*i)..*i + str.len())
                            .expect("invalid str range");
                        if self.is_within_chunk_size(chunk) {
                            true
                        } else {
                            peek_start = None;
                            false
                        }
                    })
                    .fold::<(Option<usize>, usize), _>((None, 0), |(start, _), (i, str)| {
                        (start.or(Some(i)), i + str.len())
                    });
                start.and_then(|start| text.get(start..end).map(|t| (start, t)))
            })
            // Filter out any chunks who got through as empty strings
            .filter(|(_, t)| !t.is_empty())
    }

    /// Generate iter of str indices from a regex separator. These won't be
    /// batched yet in case further fallbacks are needed.
    fn str_indices_from_regex_separator<'a, 'b: 'a>(
        text: &'b str,
        separator: &'a Regex,
    ) -> impl Iterator<Item = (usize, &'b str)> + 'a {
        let mut cursor = 0;
        let mut final_match = false;
        separator
            .find_iter(text)
            .batching(move |it| match it.next() {
                // If we've hit the end, actually return None
                None if final_match => None,
                // First time we hit None, return the final section of the text
                None => {
                    final_match = true;
                    text.get(cursor..).map(|t| Either::Left(once((cursor, t))))
                }
                // Return text preceding match + the match
                Some(sep) => {
                    let sep_range = sep.range();
                    let prev_word = (
                        cursor,
                        text.get(cursor..sep_range.start)
                            .expect("invalid character sequence in regex"),
                    );
                    let separator = (
                        sep_range.start,
                        text.get(sep_range.start..sep_range.end)
                            .expect("invalid character sequence in regex"),
                    );
                    cursor = sep_range.end;
                    Some(Either::Right([prev_word, separator].into_iter()))
                }
            })
            .flatten()
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

    /// Preserve Unicode graphemes where possible. Char iter would break them
    /// up by default.
    fn chunk_by_grapheme_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = (usize, &'b str)> + 'a {
        self.generate_chunks_from_str_indices(
            text,
            text.grapheme_indices(true).flat_map(|(i, grapheme)| {
                // If grapheme is too large, do char chunking
                if self.is_within_chunk_size(grapheme) {
                    Either::Left(once((i, grapheme)))
                } else {
                    Either::Right(
                        self.chunk_by_char_indices(grapheme)
                            // Offset relative indices back to parent string
                            .map(move |(ci, c)| (ci + i, c)),
                    )
                }
            }),
        )
    }

    /// Generate a list of chunks from a given text. Each chunk will be up to
    /// the `max_chunk_size`.
    ///
    /// If a text is too large, each chunk will fit as many
    /// [unicode graphemes](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
    /// as possible.
    ///
    /// If a given grapheme is larger than your chunk size, given the length
    /// function, then it will be passed through
    /// [`TextSplitter::chunk_by_chars`] until it will fit in a chunk.
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

    /// Preserve Unicode words wherever possible. Fallsback to graphemes if
    /// the word is larger than a chunk
    fn chunk_by_word_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = (usize, &'b str)> + 'a {
        self.generate_chunks_from_str_indices(
            text,
            text.split_word_bound_indices().flat_map(|(i, word)| {
                // If words is too large, do grapheme chunking
                if self.is_within_chunk_size(word) {
                    Either::Left(once((i, word)))
                } else {
                    Either::Right(
                        self.chunk_by_grapheme_indices(word)
                            // Offset relative indices back to parent string
                            .map(move |(gi, g)| (gi + i, g)),
                    )
                }
            }),
        )
    }

    /// Generate a list of chunks from a given text. Each chunk will be up to
    /// the `max_chunk_size`.
    ///
    /// If a text is too large, each chunk will fit as many
    /// [unicode words](https://www.unicode.org/reports/tr29/#Word_Boundaries)
    /// as possible.
    ///
    /// If a given word is larger than your chunk size, given the length
    /// function, then it will be passed through
    /// [`TextSplitter::chunk_by_graphemes`] until it will fit in a chunk.
    ///
    /// ```
    /// use text_splitter::TextSplitter;
    ///
    /// let splitter = TextSplitter::new(100);
    /// let text = "Some text from a document";
    /// let chunks = splitter.chunk_by_words(text);
    /// ```
    pub fn chunk_by_words<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = &'b str> + 'a {
        self.chunk_by_word_indices(text).map(|(_, t)| t)
    }

    /// Preserve Unicode sentences wherever possible. Fallsback to words if
    /// the word is larger than a chunk
    fn chunk_by_sentence_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = (usize, &'b str)> + 'a {
        self.generate_chunks_from_str_indices(
            text,
            text.split_sentence_bound_indices()
                .flat_map(|(i, sentence)| {
                    // If sentence is too large, do word chunking
                    if self.is_within_chunk_size(sentence) {
                        Either::Left(once((i, sentence)))
                    } else {
                        Either::Right(
                            self.chunk_by_word_indices(sentence)
                                // Offset relative indices back to parent string
                                .map(move |(wi, w)| (wi + i, w)),
                        )
                    }
                }),
        )
    }

    /// Generate a list of chunks from a given text. Each chunk will be up to
    /// the `max_chunk_size`.
    ///
    /// If a text is too large, each chunk will fit as many
    /// [unicode sentences](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
    /// as possible.
    ///
    /// If a given sentence is larger than your chunk size, given the length
    /// function, then it will be passed through
    /// [`TextSplitter::chunk_by_words`] until it will fit in a chunk.
    ///
    /// ```
    /// use text_splitter::TextSplitter;
    ///
    /// let splitter = TextSplitter::new(100);
    /// let text = "Some text from a document";
    /// let chunks = splitter.chunk_by_sentences(text);
    /// ```
    pub fn chunk_by_sentences<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = &'b str> + 'a {
        self.chunk_by_sentence_indices(text).map(|(_, t)| t)
    }

    /// Preserve Unicode sentences wherever possible. Fallsback to words if
    /// the word is larger than a chunk
    fn chunk_by_paragraph_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = (usize, &'b str)> + 'a {
        self.generate_chunks_from_str_indices(
            text,
            Self::str_indices_from_regex_separator(text, &DOUBLE_NEWLINE)
                .flat_map(|(i, paragraph)| {
                    // If paragraph is too large, do single line
                    if self.is_within_chunk_size(paragraph) {
                        Either::Left(once((i, paragraph)))
                    } else {
                        Either::Right(
                            Self::str_indices_from_regex_separator(paragraph, &NEWLINE)
                                // Offset relative indices back to parent string
                                .map(move |(pi, p)| (pi + i, p)),
                        )
                    }
                })
                .flat_map(|(i, paragraph)| {
                    // If paragraph is still too large, do sentences
                    if self.is_within_chunk_size(paragraph) {
                        Either::Left(once((i, paragraph)))
                    } else {
                        Either::Right(
                            self.chunk_by_sentence_indices(paragraph)
                                // Offset relative indices back to parent string
                                .map(move |(si, s)| (si + i, s)),
                        )
                    }
                }),
        )
    }

    /// Generate a list of chunks from a given text. Each chunk will be up to
    /// the `max_chunk_size`.
    ///
    /// If a text is too large, each chunk will fit as many paragraphs as
    /// possible, first splitting by two or more newlines (checking for both \r
    /// and \n), and then by single newlines.
    ///
    /// If a given paragraph is larger than your chunk size, given the length
    /// function, then it will be passed through
    /// [`TextSplitter::chunk_by_sentences`] until it will fit in a chunk.
    ///
    /// ```
    /// use text_splitter::TextSplitter;
    ///
    /// let splitter = TextSplitter::new(100);
    /// let text = "Some text from a document";
    /// let chunks = splitter.chunk_by_paragraphs(text);
    /// ```
    pub fn chunk_by_paragraphs<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = &'b str> + 'a {
        self.chunk_by_paragraph_indices(text).map(|(_, t)| t)
    }
}

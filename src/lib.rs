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

use core::iter::once;

use either::Either;
use itertools::Itertools;
use once_cell::sync::Lazy;
use regex::Regex;
use unicode_segmentation::UnicodeSegmentation;

mod characters;
#[cfg(feature = "huggingface-tokenizers")]
mod huggingface_tokenizers;
mod tokenizer;

pub use characters::Characters;
pub use tokenizer::{NumTokens, Tokens};

/// Determines if a given piece of text is still a valid chunk.
pub trait ChunkSize {
    /// Determine if the given chunk still fits within the specified max chunk
    /// size.
    fn valid_chunk(&self, chunk: &str) -> bool;
}

/// Default plain-text splitter. Recursively splits chunks into the smallest
/// semantic units that fit within the chunk size. Also will attempt to merge
/// neighboring chunks if they can fit within the given chunk size.
#[derive(Debug)]
pub struct TextSplitter<C>
where
    C: ChunkSize,
{
    /// Method of determining chunk sizes.
    chunk_size: C,
    /// Whether or not all chunks should have whitespace trimmed.
    /// If `false`, joining all chunks should return the original string.
    /// If `true`, all chunks will have whitespace removed from beginning and end.
    trim_chunks: bool,
}

// Lazy's so that we don't have to compile them more than once
/// Any sequence of 2 or more newlines
static DOUBLE_NEWLINE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\r\n){2,}|\r{2,}|\n{2,}").unwrap());
/// Fallback for anything else
static NEWLINE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\r|\n)+").unwrap());

impl<C> TextSplitter<C>
where
    C: ChunkSize,
{
    /// Creates a new [`TextSplitter`].
    ///
    /// ```
    /// use text_splitter::{Characters, TextSplitter};
    ///
    /// let splitter = TextSplitter::new(Characters::new(100));
    /// ```
    #[must_use]
    pub fn new(chunk_size: C) -> Self {
        Self {
            chunk_size,
            trim_chunks: false,
        }
    }

    /// Specify whether chunks should have whitespace trimmed from the
    /// beginning and end or not.
    ///
    /// If `false` (default), joining all chunks should return the original
    /// string.
    /// If `true`, all chunks will have whitespace removed from beginning and end.
    ///
    /// ```
    /// use text_splitter::{Characters, TextSplitter};
    ///
    /// let splitter = TextSplitter::new(Characters::new(100)).with_trim_chunks(true);
    /// ```
    #[must_use]
    pub fn with_trim_chunks(mut self, trim_chunks: bool) -> Self {
        self.trim_chunks = trim_chunks;
        self
    }

    /// Is the given text within the chunk size?
    fn is_within_chunk_size(&self, chunk: &str) -> bool {
        self.chunk_size.valid_chunk(if self.trim_chunks {
            chunk.trim()
        } else {
            chunk
        })
    }

    /// Internal method to handle chunk splitting for anything above char level
    fn generate_chunks_from_str_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
        it: impl Iterator<Item = (usize, &'b str)> + 'a,
    ) -> impl Iterator<Item = (usize, &'b str)> + 'a {
        it.peekable().batching(move |it| {
            let (mut start, mut end) = (None, 0);

            // Consume as many other chunks as we can
            while let Some((i, str)) = it.peek() {
                let chunk = text
                    .get(*start.get_or_insert(*i)..*i + str.len())
                    .expect("invalid str range");

                // If this doesn't fit, as long as it isn't our first one,
                // end the check here, we have a chunk.
                if !self.is_within_chunk_size(chunk) && end != 0 {
                    break;
                }

                end = i + str.len();
                it.next();
            }

            let start = start?;
            let chunk = text.get(start..end)?;
            // Trim whitespace if user requested it
            let (start, chunk) = if self.trim_chunks {
                // Figure out how many bytes we lose trimming the beginning
                let offset = chunk.len() - chunk.trim_start().len();
                (start + offset, chunk.trim())
            } else {
                (start, chunk)
            };

            // Filter out any chunks who got through as empty strings
            (!chunk.is_empty()).then_some((start, chunk))
        })
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
    /// use text_splitter::{Characters, TextSplitter};
    ///
    /// let splitter = TextSplitter::new(Characters::new(10));
    /// let text = "Some text from a document";
    /// let chunks = splitter.chunk_by_chars(text).collect::<Vec<_>>();;
    ///
    /// assert_eq!(vec!["Some text ", "from a doc", "ument"], chunks);
    /// ```
    pub fn chunk_by_chars<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = &'b str> + 'a {
        self.chunk_by_char_indices(text).map(|(_, t)| t)
    }

    /// Returns an iterator over the characters of the text and their byte offsets.
    /// See [`TextSplitter::chunk_by_chars()`] for more information.
    ///
    /// ```
    /// use text_splitter::{Characters, TextSplitter};
    ///
    /// let splitter = TextSplitter::new(Characters::new(10));
    /// let text = "Some text from a document";
    /// let chunks = splitter.chunk_by_char_indices(text).collect::<Vec<_>>();;
    ///
    /// assert_eq!(vec![(0, "Some text "), (10, "from a doc"), (20, "ument")], chunks);
    /// ```
    pub fn chunk_by_char_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = (usize, &'b str)> + 'a {
        self.generate_chunks_from_str_indices(
            text,
            text.char_indices().map(|(i, c)| {
                (
                    i,
                    text.get(i..i + c.len_utf8()).expect("char should be valid"),
                )
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
    /// use text_splitter::{Characters, TextSplitter};
    ///
    /// let splitter = TextSplitter::new(Characters::new(10));
    /// let text = "Some text\r\nfrom a document";
    /// let chunks = splitter.chunk_by_graphemes(text).collect::<Vec<_>>();;
    ///
    /// assert_eq!(vec!["Some text", "\r\nfrom a d", "ocument"], chunks);
    /// ```
    pub fn chunk_by_graphemes<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = &'b str> + 'a {
        self.chunk_by_grapheme_indices(text).map(|(_, t)| t)
    }

    /// Returns an iterator over the grapheme clusters of the text and their byte offsets.
    /// See [`TextSplitter::chunk_by_graphemes()`] for more information.
    ///
    /// ```
    /// use text_splitter::{Characters, TextSplitter};
    ///
    /// let splitter = TextSplitter::new(Characters::new(10));
    /// let text = "Some text\r\nfrom a document";
    /// let chunks = splitter.chunk_by_grapheme_indices(text).collect::<Vec<_>>();;
    ///
    /// assert_eq!(vec![(0, "Some text"), (9, "\r\nfrom a d"), (19, "ocument")], chunks);
    /// ```
    pub fn chunk_by_grapheme_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = (usize, &'b str)> + 'a {
        self.generate_chunks_from_str_indices(
            text,
            text.grapheme_indices(true).flat_map(|(i, grapheme)| {
                if self.is_within_chunk_size(grapheme) {
                    Either::Left(once((i, grapheme)))
                } else {
                    // If grapheme is too large, do char chunking
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
    /// [unicode words](https://www.unicode.org/reports/tr29/#Word_Boundaries)
    /// as possible.
    ///
    /// If a given word is larger than your chunk size, given the length
    /// function, then it will be passed through
    /// [`TextSplitter::chunk_by_graphemes`] until it will fit in a chunk.
    ///
    /// ```
    /// use text_splitter::{Characters, TextSplitter};
    ///
    /// let splitter = TextSplitter::new(Characters::new(10));
    /// let text = "Some text from a document";
    /// let chunks = splitter.chunk_by_words(text).collect::<Vec<_>>();;
    ///
    /// assert_eq!(vec!["Some text ", "from a ", "document"], chunks);
    /// ```
    pub fn chunk_by_words<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = &'b str> + 'a {
        self.chunk_by_word_indices(text).map(|(_, t)| t)
    }

    /// Returns an iterator over the words of the text and their byte offsets.
    /// See [`TextSplitter::chunk_by_words()`] for more information.
    ///
    /// ```
    /// use text_splitter::{Characters, TextSplitter};
    ///
    /// let splitter = TextSplitter::new(Characters::new(10));
    /// let text = "Some text from a document";
    /// let chunks = splitter.chunk_by_word_indices(text).collect::<Vec<_>>();;
    ///
    /// assert_eq!(vec![(0, "Some text "), (10, "from a "), (17, "document")], chunks);
    /// ```
    pub fn chunk_by_word_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = (usize, &'b str)> + 'a {
        self.generate_chunks_from_str_indices(
            text,
            text.split_word_bound_indices().flat_map(|(i, word)| {
                if self.is_within_chunk_size(word) {
                    Either::Left(once((i, word)))
                } else {
                    // If words is too large, do grapheme chunking
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
    /// [unicode sentences](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
    /// as possible.
    ///
    /// If a given sentence is larger than your chunk size, given the length
    /// function, then it will be passed through
    /// [`TextSplitter::chunk_by_words`] until it will fit in a chunk.
    ///
    /// ```
    /// use text_splitter::{Characters, TextSplitter};
    ///
    /// let splitter = TextSplitter::new(Characters::new(10));
    /// let text = "Some text. From a document.";
    /// let chunks = splitter.chunk_by_sentences(text).collect::<Vec<_>>();;
    ///
    /// assert_eq!(vec!["Some text.", " From a ", "document."], chunks);
    /// ```
    pub fn chunk_by_sentences<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = &'b str> + 'a {
        self.chunk_by_sentence_indices(text).map(|(_, t)| t)
    }

    /// Returns an iterator over the unicode sentences of the text and their byte offsets.
    /// See [`TextSplitter::chunk_by_sentences()`] for more information.
    ///
    /// ```
    /// use text_splitter::{Characters, TextSplitter};
    ///
    /// let splitter = TextSplitter::new(Characters::new(10));
    /// let text = "Some text. From a document.";
    /// let chunks = splitter.chunk_by_sentence_indices(text).collect::<Vec<_>>();;
    ///
    /// assert_eq!(vec![(0, "Some text."), (10, " From a "), (18, "document.")], chunks);
    /// ```
    pub fn chunk_by_sentence_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = (usize, &'b str)> + 'a {
        self.generate_chunks_from_str_indices(
            text,
            text.split_sentence_bound_indices()
                .flat_map(|(i, sentence)| {
                    if self.is_within_chunk_size(sentence) {
                        Either::Left(once((i, sentence)))
                    } else {
                        // If sentence is too large, do word chunking
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
    /// If a text is too large, each chunk will fit as many paragraphs as
    /// possible, first splitting by two or more newlines (checking for both \r
    /// and \n), and then by single newlines.
    ///
    /// If a given paragraph is larger than your chunk size, given the length
    /// function, then it will be passed through
    /// [`TextSplitter::chunk_by_sentences`] until it will fit in a chunk.
    ///
    /// ```
    /// use text_splitter::{Characters, TextSplitter};
    ///
    /// let splitter = TextSplitter::new(Characters::new(10));
    /// let text = "Some text\n\nfrom a\ndocument";
    /// let chunks = splitter.chunk_by_paragraphs(text).collect::<Vec<_>>();;
    ///
    /// assert_eq!(vec!["Some text", "\n\nfrom a\n", "document"], chunks);
    /// ```
    pub fn chunk_by_paragraphs<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = &'b str> + 'a {
        self.chunk_by_paragraph_indices(text).map(|(_, t)| t)
    }

    /// Returns an iterator over the paragraphs of the text and their byte offsets.
    /// See [`TextSplitter::chunk_by_paragraphs()`] for more information.
    ///
    /// ```
    /// use text_splitter::{Characters, TextSplitter};
    ///
    /// let splitter = TextSplitter::new(Characters::new(10));
    /// let text = "Some text\n\nfrom a\ndocument";
    /// let chunks = splitter.chunk_by_paragraph_indices(text).collect::<Vec<_>>();;
    ///
    /// assert_eq!(vec![(0, "Some text"), (9, "\n\nfrom a\n"), (18, "document")], chunks);
    /// ```
    pub fn chunk_by_paragraph_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = (usize, &'b str)> + 'a {
        self.generate_chunks_from_str_indices(
            text,
            Self::str_indices_from_regex_separator(text, &DOUBLE_NEWLINE)
                .flat_map(|(i, paragraph)| {
                    if self.is_within_chunk_size(paragraph) {
                        Either::Left(once((i, paragraph)))
                    } else {
                        // If paragraph is too large, do single line
                        Either::Right(
                            Self::str_indices_from_regex_separator(paragraph, &NEWLINE)
                                // Offset relative indices back to parent string
                                .map(move |(pi, p)| (pi + i, p)),
                        )
                    }
                })
                .flat_map(|(i, paragraph)| {
                    if self.is_within_chunk_size(paragraph) {
                        Either::Left(once((i, paragraph)))
                    } else {
                        // If paragraph is still too large, do sentences
                        Either::Right(
                            self.chunk_by_sentence_indices(paragraph)
                                // Offset relative indices back to parent string
                                .map(move |(si, s)| (si + i, s)),
                        )
                    }
                }),
        )
    }
}

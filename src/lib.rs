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
use thiserror::Error;
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Error)]
/// Possible errors that can be generated when splitting text
pub enum TextSplitterError {
    /// Result of a failed check by the length function. Most likely due to
    /// failed tokenization or something similar.
    #[error("Failed to check length. Error: {source} for chunk: {chunk}")]
    LengthCheck {
        /// The chunk that the length check failed on
        chunk: String,
        /// Original error from the legnth function
        source: anyhow::Error,
    },
}

type LengthFn = dyn Fn(&str) -> anyhow::Result<usize>;

/// Default plain-text splitter. Recursively splits chunks into the smallest
/// semantic units that fit within the chunk size. Also will attempt to merge
/// neighboring chunks if they can fit within the given chunk size.
pub struct TextSplitter {
    /// Method of calculating chunk length. By default done at the character level.
    length_fn: Box<LengthFn>,
    /// Maximum size of a chunk (measured by length_fn)
    max_chunk_size: usize,
    /// Whether or not all chunks should have whitespace trimmed.
    /// If `false`, joining all chunks should return the original string.
    /// If `true`, all chunks will have whitespace removed from beginning and end.
    trim_chunks: bool,
}

impl fmt::Debug for TextSplitter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TextSplitter")
            .field("max_chunk_size", &self.max_chunk_size)
            .field("trim_chunks", &self.trim_chunks)
            .finish()
    }
}

// Lazy's so that we don't have to compile them more than once
/// Any sequence of 2 or more newlines
static DOUBLE_NEWLINE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\r\n){2,}|\r{2,}|\n{2,}").unwrap());
/// Fallback for anything else
static NEWLINE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\r|\n)+").unwrap());

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
            length_fn: Box::new(|text| Ok(text.chars().count())),
            max_chunk_size,
            trim_chunks: false,
        }
    }

    /// Specify a custom function for calculating the length of a chunk. For
    /// example, using chars instead of bytes.
    ///
    /// If `false` (default), joining all chunks should return the original
    /// string.
    /// If `true`, all chunks will have whitespace removed from beginning and end.
    ///
    /// ```
    /// use text_splitter::TextSplitter;
    ///
    /// let splitter = TextSplitter::new(100).with_trim_chunks(true);
    /// ```
    #[must_use]
    pub fn with_trim_chunks(mut self, trim_chunks: bool) -> Self {
        self.trim_chunks = trim_chunks;
        self
    }

    /// Specify whether chunks should have whitespace trimmed from the
    /// beginning and end or not.
    ///
    /// ```
    /// use text_splitter::TextSplitter;
    ///
    /// let splitter = TextSplitter::new(100).with_length_fn(|text| Ok(text.chars().count()));
    /// ```
    #[must_use]
    pub fn with_length_fn(
        mut self,
        length_fn: impl Fn(&str) -> anyhow::Result<usize> + 'static,
    ) -> Self {
        self.length_fn = Box::new(length_fn);
        self
    }

    /// Is the given text within the chunk size?
    fn is_within_chunk_size(&self, chunk: &str) -> Result<bool, TextSplitterError> {
        let chunk = if self.trim_chunks {
            chunk.trim()
        } else {
            chunk
        };

        let length = (self.length_fn)(chunk).map_err(|source| TextSplitterError::LengthCheck {
            chunk: chunk.to_owned(),
            source,
        })?;

        Ok(length <= self.max_chunk_size)
    }

    /// Internal method to handle chunk splitting for anything above char level
    fn generate_chunks_from_str_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
        it: impl Iterator<Item = Result<(usize, &'b str), TextSplitterError>> + 'a,
    ) -> impl Iterator<Item = Result<(usize, &'b str), TextSplitterError>> + 'a {
        it.peekable().batching(move |it| {
            let (mut start, mut end) = (None, 0);

            // Bubble up errors
            if let Some(Err(_)) = it.peek() {
                return it.next();
            }

            // Consume as many other chunks as we can
            while let Some(Ok((i, str))) = it.peek() {
                let chunk = text
                    .get(*start.get_or_insert(*i)..*i + str.len())
                    .expect("invalid str range");
                match self.is_within_chunk_size(chunk) {
                    // If this doesn't fit or errors, as long as it isn't our first one,
                    // end the check here, we have a chunk.
                    Ok(false) | Err(_) if end != 0 => {
                        break;
                    }
                    // If this is an error, but our first chunk, we can't process it.
                    // Consume it and bail.
                    Err(e) => {
                        it.next();
                        return Some(Err(e));
                    }
                    _ => {}
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
            (!chunk.is_empty()).then_some(Ok((start, chunk)))
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
    /// use text_splitter::TextSplitter;
    ///
    /// let splitter = TextSplitter::new(10);
    /// let text = "Some text from a document";
    /// let chunks = splitter.chunk_by_chars(text).collect::<Result<Vec<_>, _>>().unwrap();
    ///
    /// assert_eq!(vec!["Some text ", "from a doc", "ument"], chunks);
    /// ```
    pub fn chunk_by_chars<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = Result<&'b str, TextSplitterError>> + 'a {
        self.chunk_by_char_indices(text).map_ok(|(_, t)| t)
    }

    /// Returns an iterator over the characters of the text and their byte offsets.
    /// See [`TextSplitter::chunk_by_chars()`] for more information.
    ///
    /// ```
    /// use text_splitter::TextSplitter;
    ///
    /// let splitter = TextSplitter::new(10);
    /// let text = "Some text from a document";
    /// let chunks = splitter.chunk_by_char_indices(text).collect::<Result<Vec<_>, _>>().unwrap();
    ///
    /// assert_eq!(vec![(0, "Some text "), (10, "from a doc"), (20, "ument")], chunks);
    /// ```
    pub fn chunk_by_char_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = Result<(usize, &'b str), TextSplitterError>> + 'a {
        self.generate_chunks_from_str_indices(
            text,
            text.char_indices().map(|(i, c)| {
                Ok((
                    i,
                    text.get(i..i + c.len_utf8()).expect("char should be valid"),
                ))
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
    /// let splitter = TextSplitter::new(10);
    /// let text = "Some text\r\nfrom a document";
    /// let chunks = splitter.chunk_by_graphemes(text).collect::<Result<Vec<_>, _>>().unwrap();
    ///
    /// assert_eq!(vec!["Some text", "\r\nfrom a d", "ocument"], chunks);
    /// ```
    pub fn chunk_by_graphemes<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = Result<&'b str, TextSplitterError>> + 'a {
        self.chunk_by_grapheme_indices(text).map_ok(|(_, t)| t)
    }

    /// Returns an iterator over the grapheme clusters of the text and their byte offsets.
    /// See [`TextSplitter::chunk_by_graphemes()`] for more information.
    ///
    /// ```
    /// use text_splitter::TextSplitter;
    ///
    /// let splitter = TextSplitter::new(10);
    /// let text = "Some text\r\nfrom a document";
    /// let chunks = splitter.chunk_by_grapheme_indices(text).collect::<Result<Vec<_>, _>>().unwrap();
    ///
    /// assert_eq!(vec![(0, "Some text"), (9, "\r\nfrom a d"), (19, "ocument")], chunks);
    /// ```
    pub fn chunk_by_grapheme_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = Result<(usize, &'b str), TextSplitterError>> + 'a {
        self.generate_chunks_from_str_indices(
            text,
            text.grapheme_indices(true).flat_map(|(i, grapheme)| {
                match self.is_within_chunk_size(grapheme) {
                    Ok(true) => Either::Left(once(Ok((i, grapheme)))),
                    // If grapheme is too large, do char chunking
                    Ok(false) => Either::Right(
                        self.chunk_by_char_indices(grapheme)
                            // Offset relative indices back to parent string
                            .map_ok(move |(ci, c)| (ci + i, c)),
                    ),
                    Err(e) => Either::Left(once(Err(e))),
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
    /// let splitter = TextSplitter::new(10);
    /// let text = "Some text from a document";
    /// let chunks = splitter.chunk_by_words(text).collect::<Result<Vec<_>, _>>().unwrap();
    ///
    /// assert_eq!(vec!["Some text ", "from a ", "document"], chunks);
    /// ```
    pub fn chunk_by_words<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = Result<&'b str, TextSplitterError>> + 'a {
        self.chunk_by_word_indices(text).map_ok(|(_, t)| t)
    }

    /// Returns an iterator over the words of the text and their byte offsets.
    /// See [`TextSplitter::chunk_by_words()`] for more information.
    ///
    /// ```
    /// use text_splitter::TextSplitter;
    ///
    /// let splitter = TextSplitter::new(10);
    /// let text = "Some text from a document";
    /// let chunks = splitter.chunk_by_word_indices(text).collect::<Result<Vec<_>, _>>().unwrap();
    ///
    /// assert_eq!(vec![(0, "Some text "), (10, "from a "), (17, "document")], chunks);
    /// ```
    pub fn chunk_by_word_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = Result<(usize, &'b str), TextSplitterError>> + 'a {
        self.generate_chunks_from_str_indices(
            text,
            text.split_word_bound_indices().flat_map(|(i, word)| {
                match self.is_within_chunk_size(word) {
                    Ok(true) => Either::Left(once(Ok((i, word)))),
                    // If words is too large, do grapheme chunking
                    Ok(false) => Either::Right(
                        self.chunk_by_grapheme_indices(word)
                            // Offset relative indices back to parent string
                            .map_ok(move |(gi, g)| (gi + i, g)),
                    ),
                    Err(e) => Either::Left(once(Err(e))),
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
    /// let splitter = TextSplitter::new(10);
    /// let text = "Some text. From a document.";
    /// let chunks = splitter.chunk_by_sentences(text).collect::<Result<Vec<_>, _>>().unwrap();
    ///
    /// assert_eq!(vec!["Some text.", " From a ", "document."], chunks);
    /// ```
    pub fn chunk_by_sentences<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = Result<&'b str, TextSplitterError>> + 'a {
        self.chunk_by_sentence_indices(text).map_ok(|(_, t)| t)
    }

    /// Returns an iterator over the unicode sentences of the text and their byte offsets.
    /// See [`TextSplitter::chunk_by_sentences()`] for more information.
    ///
    /// ```
    /// use text_splitter::TextSplitter;
    ///
    /// let splitter = TextSplitter::new(10);
    /// let text = "Some text. From a document.";
    /// let chunks = splitter.chunk_by_sentence_indices(text).collect::<Result<Vec<_>, _>>().unwrap();
    ///
    /// assert_eq!(vec![(0, "Some text."), (10, " From a "), (18, "document.")], chunks);
    /// ```
    pub fn chunk_by_sentence_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = Result<(usize, &'b str), TextSplitterError>> + 'a {
        self.generate_chunks_from_str_indices(
            text,
            text.split_sentence_bound_indices()
                .flat_map(|(i, sentence)| {
                    match self.is_within_chunk_size(sentence) {
                        Ok(true) => Either::Left(once(Ok((i, sentence)))),
                        // If sentence is too large, do word chunking
                        Ok(false) => Either::Right(
                            self.chunk_by_word_indices(sentence)
                                // Offset relative indices back to parent string
                                .map_ok(move |(wi, w)| (wi + i, w)),
                        ),
                        Err(e) => Either::Left(once(Err(e))),
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
    /// let splitter = TextSplitter::new(10);
    /// let text = "Some text\n\nfrom a\ndocument";
    /// let chunks = splitter.chunk_by_paragraphs(text).collect::<Result<Vec<_>, _>>().unwrap();
    ///
    /// assert_eq!(vec!["Some text", "\n\nfrom a\n", "document"], chunks);
    /// ```
    pub fn chunk_by_paragraphs<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = Result<&'b str, TextSplitterError>> + 'a {
        self.chunk_by_paragraph_indices(text).map_ok(|(_, t)| t)
    }

    /// Returns an iterator over the paragraphs of the text and their byte offsets.
    /// See [`TextSplitter::chunk_by_paragraphs()`] for more information.
    ///
    /// ```
    /// use text_splitter::TextSplitter;
    ///
    /// let splitter = TextSplitter::new(10);
    /// let text = "Some text\n\nfrom a\ndocument";
    /// let chunks = splitter.chunk_by_paragraph_indices(text).collect::<Result<Vec<_>, _>>().unwrap();
    ///
    /// assert_eq!(vec![(0, "Some text"), (9, "\n\nfrom a\n"), (18, "document")], chunks);
    /// ```
    pub fn chunk_by_paragraph_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
    ) -> impl Iterator<Item = Result<(usize, &'b str), TextSplitterError>> + 'a {
        self.generate_chunks_from_str_indices(
            text,
            Self::str_indices_from_regex_separator(text, &DOUBLE_NEWLINE)
                .flat_map(|(i, paragraph)| {
                    match self.is_within_chunk_size(paragraph) {
                        Ok(true) => Either::Left(once(Ok((i, paragraph)))),
                        // If paragraph is too large, do single line
                        Ok(false) => Either::Right(
                            Self::str_indices_from_regex_separator(paragraph, &NEWLINE)
                                // Offset relative indices back to parent string
                                .map(move |(pi, p)| Ok((pi + i, p))),
                        ),
                        Err(e) => Either::Left(once(Err(e))),
                    }
                })
                .flat_map(|result| {
                    match result {
                        Ok((i, paragraph)) => {
                            Either::Left(match self.is_within_chunk_size(paragraph) {
                                Ok(true) => Either::Left(once(Ok((i, paragraph)))),
                                // If paragraph is still too large, do sentences
                                Ok(false) => Either::Right(
                                    self.chunk_by_sentence_indices(paragraph)
                                        // Offset relative indices back to parent string
                                        .map_ok(move |(si, s)| (si + i, s)),
                                ),
                                Err(e) => Either::Left(once(Err(e))),
                            })
                        }
                        Err(_) => Either::Right(once(result)),
                    }
                }),
        )
    }
}

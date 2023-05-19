/*!
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
// Can also use tiktoken-rs, or anything that implements the TokenCount
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

1. 2 or more newlines (Newline is `\r\n`, `\n`, or `\r`)
2. 1 newline
3. [Unicode Sentence Boundaries](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
4. [Unicode Word Boundaries](https://www.unicode.org/reports/tr29/#Word_Boundaries)
5. [Unicode Grapheme Cluster Boundaries](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
6. Characters

Splitting doesn't occur below the character level, otherwise you could get partial bytes of a char, which may not be a valid unicode str.

_Note on sentences:_ There are lots of methods of determining sentence breaks, all to varying degrees of accuracy, and many requiring ML models to do so. Rather than trying to find the perfect sentence breaks, we rely on unicode method of sentence boundaries, which in most cases is good enough for finding a decent semantic breaking point if a paragraph is too large, and avoids the performance penalties of many other methods.

## Inspiration

This crate was inspired by [LangChain's TextSplitter](https://python.langchain.com/en/latest/modules/indexes/text_splitters/examples/recursive_text_splitter.html). But, looking into the implementation, there was potential for better performance as well as better semantic chunking.

A big thank you to the unicode-rs team for their [unicode-segmentation](https://crates.io/crates/unicode-segmentation) crate that manages a lot of the complexity of matching the Unicode rules for words and sentences.

*/

#![warn(
    clippy::cargo,
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
#![cfg_attr(docsrs, feature(doc_auto_cfg, doc_cfg))]

use core::{iter::once, ops::Range};

use auto_enums::auto_enum;
use either::Either;
use itertools::Itertools;
use once_cell::sync::Lazy;
use regex::Regex;
use unicode_segmentation::UnicodeSegmentation;

mod characters;
#[cfg(feature = "tokenizers")]
mod huggingface;
#[cfg(feature = "tiktoken-rs")]
mod tiktoken;
mod tokenizer;

pub use characters::Characters;
pub use tokenizer::TokenCount;

/// Determines if a given piece of text is still a valid chunk.
pub trait ChunkValidator {
    /// Determine if the given chunk still fits within the specified max chunk
    /// size.
    fn validate_chunk(&self, chunk: &str, chunk_size: usize) -> bool;
}

/// Default plain-text splitter. Recursively splits chunks into the largest
/// semantic units that fit within the chunk size. Also will attempt to merge
/// neighboring chunks if they can fit within the given chunk size.
#[derive(Debug)]
pub struct TextSplitter<V>
where
    V: ChunkValidator,
{
    /// Method of determining chunk sizes.
    chunk_validator: V,
    /// Whether or not all chunks should have whitespace trimmed.
    /// If `false`, joining all chunks should return the original string.
    /// If `true`, all chunks will have whitespace removed from beginning and end.
    trim_chunks: bool,
}

impl Default for TextSplitter<Characters> {
    fn default() -> Self {
        Self::new(Characters)
    }
}

impl<V> TextSplitter<V>
where
    V: ChunkValidator,
{
    /// Creates a new [`TextSplitter`].
    ///
    /// ```
    /// use text_splitter::{Characters, TextSplitter};
    ///
    /// // Characters is the default, so you can also do `TextSplitter::default()`
    /// let splitter = TextSplitter::new(Characters);
    /// ```
    #[must_use]
    pub fn new(chunk_validator: V) -> Self {
        Self {
            chunk_validator,
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
    /// let splitter = TextSplitter::default().with_trim_chunks(true);
    /// ```
    #[must_use]
    pub fn with_trim_chunks(mut self, trim_chunks: bool) -> Self {
        self.trim_chunks = trim_chunks;
        self
    }

    /// Generate a list of chunks from a given text. Each chunk will be up to
    /// the `max_chunk_size`.
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
    /// 1. 2 or more newlines (Newline is `\r\n`, `\n`, or `\r`)
    /// 2. 1 newline
    /// 3. [Unicode Sentence Boundaries](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
    /// 4. [Unicode Word Boundaries](https://www.unicode.org/reports/tr29/#Word_Boundaries)
    /// 5. [Unicode Grapheme Cluster Boundaries](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
    /// 6. Characters
    ///
    /// Splitting doesn't occur below the character level, otherwise you could get partial
    /// bytes of a char, which may not be a valid unicode str.
    ///
    /// ```
    /// use text_splitter::{Characters, TextSplitter};
    ///
    /// let splitter = TextSplitter::default();
    /// let text = "Some text\n\nfrom a\ndocument";
    /// let chunks = splitter.chunks(text, 10).collect::<Vec<_>>();
    ///
    /// assert_eq!(vec!["Some text", "\n\n", "from a\n", "document"], chunks);
    /// ```
    pub fn chunks<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
        chunk_size: usize,
    ) -> impl Iterator<Item = &'text str> + 'splitter {
        self.chunk_indices(text, chunk_size).map(|(_, t)| t)
    }

    /// Returns an iterator over chunks of the text and their byte offsets.
    /// Each chunk will be up to the `max_chunk_size`.
    ///
    /// See [`TextSplitter::chunks`] for more information.
    ///
    /// ```
    /// use text_splitter::{Characters, TextSplitter};
    ///
    /// let splitter = TextSplitter::default();
    /// let text = "Some text\n\nfrom a\ndocument";
    /// let chunks = splitter.chunk_indices(text, 10).collect::<Vec<_>>();
    ///
    /// assert_eq!(vec![(0, "Some text"), (9, "\n\n"), (11, "from a\n"), (18, "document")], chunks);
    pub fn chunk_indices<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
        chunk_size: usize,
    ) -> TextChunks<'text, 'splitter, V> {
        TextChunks::new(
            chunk_size,
            &self.chunk_validator,
            SemanticLevel::DoubleLineBreak,
            text,
            self.trim_chunks,
        )
    }
}

// Lazy so that we don't have to compile them more than once
/// Any sequence of 2 or more newlines
static DOUBLE_LINEBREAK: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(\r\n){2,}|\r{2,}|\n{2,}").unwrap());
/// Fallback for anything else
static LINEBREAK: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\r|\n)+").unwrap());

/// Different semantic levels that text can be split by.
/// Each level provides a method of splitting text into chunks of a given level
/// as well as a fallback in case a given fallback is too large.
#[derive(Clone, Copy, Debug)]
enum SemanticLevel {
    /// Split by individual chars. May be larger than a single byte,
    /// but we don't go lower so we always have valid UTF str's.
    Char,
    /// Split by [unicode grapheme clusters](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)    Grapheme,
    /// Falls back to [`Self::Char`]
    GraphemeCluster,
    /// Split by [unicode words](https://www.unicode.org/reports/tr29/#Word_Boundaries)
    /// Falls back to [`Self::GraphemeCluster`]
    Word,
    /// Split by [unicode sentences](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
    /// Falls back to [`Self::Word`]
    Sentence,
    /// Split by single linebreak, either `\n`, `\r`, or `\r\n`.
    /// Falls back to [`Self::Sentence`]
    LineBreak,
    /// Split by 2 or more linebreaks, either `\n`, `\r`, or `\r\n`.
    /// Falls back to [`Self::LineBreak`]
    DoubleLineBreak,
}

impl SemanticLevel {
    /// Optional fallback, if available, if the current level is too large
    fn fallback(self) -> Option<Self> {
        match self {
            Self::Char => None,
            Self::GraphemeCluster => Some(Self::Char),
            Self::Word => Some(Self::GraphemeCluster),
            Self::Sentence => Some(Self::Word),
            Self::LineBreak => Some(Self::Sentence),
            Self::DoubleLineBreak => Some(Self::LineBreak),
        }
    }

    /// Split a given text into iterator over each semantic chunk
    #[auto_enum(Iterator)]
    fn chunks(self, text: &str) -> impl Iterator<Item = &str> {
        match self {
            Self::Char => text
                .char_indices()
                .map(|(i, c)| text.get(i..i + c.len_utf8()).expect("char should be valid")),
            Self::GraphemeCluster => text.graphemes(true),
            Self::Word => text.split_word_bounds(),
            Self::Sentence => text.split_sentence_bounds(),
            Self::LineBreak => split_str_by_regex_separator(text, &LINEBREAK),
            Self::DoubleLineBreak => split_str_by_regex_separator(text, &DOUBLE_LINEBREAK),
        }
    }
}

/// Returns chunks of text with their byte offsets as an iterator.
#[derive(Debug)]
pub struct TextChunks<'text, 'validator, V>
where
    V: ChunkValidator,
{
    /// Size of the chunks to generate
    chunk_size: usize,
    /// How to validate chunk sizes
    chunk_validator: &'validator V,
    /// Current byte offset in the `text`
    cursor: usize,
    /// Largest Semantic Level we are splitting by
    semantic_level: SemanticLevel,
    /// Original text to iterate over and generate chunks from
    text: &'text str,
    /// Whether or not chunks should be trimmed
    trim_chunks: bool,
}

impl<'text, 'validator, V> TextChunks<'text, 'validator, V>
where
    V: ChunkValidator,
{
    /// Generate new [`TextChunks`] iterator for a given text.
    /// Starts with an offset of 0
    fn new(
        chunk_size: usize,
        chunk_validator: &'validator V,
        semantic_level: SemanticLevel,
        text: &'text str,
        trim_chunks: bool,
    ) -> Self {
        Self {
            cursor: 0,
            chunk_size,
            chunk_validator,
            semantic_level,
            text,
            trim_chunks,
        }
    }

    /// Is the given text within the chunk size?
    fn validate_chunk(&self, chunk: &str) -> bool {
        self.chunk_validator.validate_chunk(
            if self.trim_chunks {
                chunk.trim()
            } else {
                chunk
            },
            self.chunk_size,
        )
    }

    /// Generate chunk for a given range, applying trimming settings.
    /// Returns final byte offset and str.
    /// Will return `None` if given an invalid range.
    fn chunk(&self, range: Range<usize>) -> Option<(usize, &'text str)> {
        let start = range.start;
        let chunk = self.text.get(range)?;
        // Trim whitespace if user requested it
        Some(if self.trim_chunks {
            // Figure out how many bytes we lose trimming the beginning
            let offset = chunk.len() - chunk.trim_start().len();
            (start + offset, chunk.trim())
        } else {
            (start, chunk)
        })
    }
}

impl<'text, 'validator, V> Iterator for TextChunks<'text, 'validator, V>
where
    V: ChunkValidator,
{
    type Item = (usize, &'text str);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Make sure we haven't reached the end
            if self.cursor >= self.text.len() {
                return None;
            }

            // Start with the top-level breaks
            let start = self.cursor;

            let mut it = self
                .semantic_level
                .chunks(self.text.get(start..)?)
                .peekable();

            // Check if we need to split it up first
            let mut current_level = self.semantic_level;
            loop {
                match (it.peek(), current_level.fallback()) {
                    // Break early, nothing to do here
                    (None, _) => return None,
                    // If we can't break up further, exit loop
                    (Some(_), None) => break,
                    // If this is a valid chunk, we are fine
                    (Some(str), _) if self.validate_chunk(str) => {
                        break;
                    }
                    // Otherwise break up the text with the fallback
                    (Some(str), Some(fallback)) => {
                        it = fallback.chunks(str).peekable();
                        current_level = fallback;
                    }
                }
            }

            // Consume as many as we can fit
            for str in it {
                let chunk = self.text.get(start..self.cursor + str.len())?;
                // If this doesn't fit, as log as it isn't our first one, end the check here,
                // we have a chunk.
                if !self.validate_chunk(chunk) && start != self.cursor {
                    break;
                }

                self.cursor += str.len();
            }

            let (start, chunk) = self.chunk(start..self.cursor)?;

            // Make sure we didn't get an empty chunk. Should only happen in
            // cases where we trim.
            if chunk.is_empty() {
                continue;
            }

            // Return a chunk if we have one
            return Some((start, chunk));
        }
    }
}

/// Generate iter of str indices from a regex separator. These won't be
/// batched yet in case further fallbacks are needed.
fn split_str_by_regex_separator<'sep, 'text: 'sep>(
    text: &'text str,
    separator: &'sep Regex,
) -> impl Iterator<Item = &'text str> + 'sep {
    split_str_by_separator(text, true, separator.find_iter(text).map(|sep| sep.range()))
}

/// Given a list of separator ranges, construct the sections of the text
fn split_str_by_separator(
    text: &str,
    separator_is_own_chunk: bool,
    separator_ranges: impl Iterator<Item = Range<usize>>,
) -> impl Iterator<Item = &str> {
    let mut cursor = 0;
    let mut final_match = false;
    separator_ranges
        .batching(move |it| match it.next() {
            // If we've hit the end, actually return None
            None if final_match => None,
            // First time we hit None, return the final section of the text
            None => {
                final_match = true;
                text.get(cursor..).map(|t| Either::Left(once(t)))
            }
            // Return text preceding match + the match
            Some(range) if separator_is_own_chunk => {
                let prev_section = text
                    .get(cursor..range.start)
                    .expect("invalid character sequence");
                let separator = text
                    .get(range.start..range.end)
                    .expect("invalid character sequence");
                cursor = range.end;
                Some(Either::Right([prev_section, separator].into_iter()))
            }
            // Return just the text preceding the match
            Some(range) => {
                let prev_section = text
                    .get(cursor..range.start)
                    .expect("invalid character sequence");
                // Separator will be part of the next chunk
                cursor = range.start;
                Some(Either::Left(once(prev_section)))
            }
        })
        .flatten()
}

#[cfg(test)]
mod tests {
    use std::cmp::min;

    use fake::{Fake, Faker};

    use super::*;

    #[test]
    fn returns_one_chunk_if_text_is_shorter_than_max_chunk_size() {
        let text = Faker.fake::<String>();
        let chunks = TextChunks::new(
            text.chars().count(),
            &Characters,
            SemanticLevel::Char,
            &text,
            false,
        )
        .map(|(_, c)| c)
        .collect::<Vec<_>>();
        assert_eq!(vec![&text], chunks);
    }

    #[test]
    fn returns_two_chunks_if_text_is_longer_than_max_chunk_size() {
        let text1 = Faker.fake::<String>();
        let text2 = Faker.fake::<String>();
        let text = format!("{text1}{text2}");
        // Round up to one above half so it goes to 2 chunks
        let max_chunk_size = text.chars().count() / 2 + 1;

        let chunks = TextChunks::new(
            max_chunk_size,
            &Characters,
            SemanticLevel::Char,
            &text,
            false,
        )
        .map(|(_, c)| c)
        .collect::<Vec<_>>();

        assert!(chunks.iter().all(|c| c.chars().count() <= max_chunk_size));

        // Check that beginning of first chunk and text 1 matches
        let len = min(text1.len(), chunks[0].len());
        assert_eq!(text1[..len], chunks[0][..len]);
        // Check that end of second chunk and text 2 matches
        let len = min(text2.len(), chunks[1].len());
        assert_eq!(
            text2[(text2.len() - len)..],
            chunks[1][chunks[1].len() - len..]
        );

        assert_eq!(chunks.join(""), text);
    }

    #[test]
    fn empty_string() {
        let text = "";
        let chunks = TextChunks::new(100, &Characters, SemanticLevel::Char, text, false)
            .map(|(_, c)| c)
            .collect::<Vec<_>>();
        assert!(chunks.is_empty());
    }

    #[test]
    fn can_handle_unicode_characters() {
        let text = "éé"; // Char that is more than one byte
        let chunks = TextChunks::new(1, &Characters, SemanticLevel::Char, text, false)
            .map(|(_, c)| c)
            .collect::<Vec<_>>();
        assert_eq!(vec!["é", "é"], chunks);
    }

    // Just for testing
    struct Str;

    impl ChunkValidator for Str {
        fn validate_chunk(&self, chunk: &str, chunk_size: usize) -> bool {
            chunk.len() <= chunk_size
        }
    }

    #[test]
    fn custom_len_function() {
        let text = "éé"; // Char that is two bytes each
        let chunks = TextChunks::new(2, &Str, SemanticLevel::Char, text, false)
            .map(|(_, c)| c)
            .collect::<Vec<_>>();
        assert_eq!(vec!["é", "é"], chunks);
    }

    #[test]
    fn handles_char_bigger_than_len() {
        let text = "éé"; // Char that is two bytes each
        let chunks = TextChunks::new(1, &Str, SemanticLevel::Char, text, false)
            .map(|(_, c)| c)
            .collect::<Vec<_>>();
        // We can only go so small
        assert_eq!(vec!["é", "é"], chunks);
    }

    #[test]
    fn chunk_by_graphemes() {
        let text = "a̐éö̲\r\n";

        let chunks = TextChunks::new(3, &Characters, SemanticLevel::GraphemeCluster, text, false)
            .map(|(_, g)| g)
            .collect::<Vec<_>>();
        // \r\n is grouped together not separated
        assert_eq!(vec!["a̐é", "ö̲", "\r\n"], chunks);
    }

    #[test]
    fn trim_char_indices() {
        let text = " a b ";

        let chunks =
            TextChunks::new(1, &Characters, SemanticLevel::Char, text, true).collect::<Vec<_>>();
        assert_eq!(vec![(1, "a"), (3, "b")], chunks);
    }

    #[test]
    fn graphemes_fallback_to_chars() {
        let text = "a̐éö̲\r\n";

        let chunks = TextChunks::new(1, &Characters, SemanticLevel::GraphemeCluster, text, false)
            .map(|(_, g)| g)
            .collect::<Vec<_>>();
        assert_eq!(
            vec!["a", "\u{310}", "é", "ö", "\u{332}", "\r", "\n"],
            chunks
        );
    }

    #[test]
    fn trim_grapheme_indices() {
        let text = "\r\na̐éö̲\r\n";

        let chunks = TextChunks::new(3, &Characters, SemanticLevel::GraphemeCluster, text, true)
            .collect::<Vec<_>>();
        assert_eq!(vec![(2, "a̐é"), (7, "ö̲")], chunks);
    }

    #[test]
    fn chunk_by_words() {
        let text = "The quick (\"brown\") fox can't jump 32.3 feet, right?";

        let chunks = TextChunks::new(10, &Characters, SemanticLevel::Word, text, false)
            .map(|(_, w)| w)
            .collect::<Vec<_>>();
        assert_eq!(
            vec![
                "The quick ",
                "(\"brown\") ",
                "fox can't ",
                "jump 32.3 ",
                "feet, ",
                "right?"
            ],
            chunks
        );
    }

    #[test]
    fn words_fallback_to_graphemes() {
        let text = "Thé quick\r\n";
        let chunks = TextChunks::new(2, &Characters, SemanticLevel::Word, text, false)
            .map(|(_, w)| w)
            .collect::<Vec<_>>();
        assert_eq!(vec!["Th", "é ", "qu", "ic", "k", "\r\n"], chunks);
    }

    #[test]
    fn trim_word_indices() {
        let text = "Some text from a document";
        let chunks =
            TextChunks::new(10, &Characters, SemanticLevel::Word, text, true).collect::<Vec<_>>();
        assert_eq!(
            vec![(0, "Some text"), (10, "from a"), (17, "document")],
            chunks
        );
    }

    #[test]
    fn chunk_by_sentences() {
        let text = "Mr. Fox jumped. [...] The dog was too lazy.";
        let chunks = TextChunks::new(21, &Characters, SemanticLevel::Sentence, text, false)
            .map(|(_, s)| s)
            .collect::<Vec<_>>();
        assert_eq!(
            vec!["Mr. Fox jumped. ", "[...] ", "The dog was too lazy."],
            chunks
        );
    }

    #[test]
    fn sentences_falls_back_to_words() {
        let text = "Mr. Fox jumped. [...] The dog was too lazy.";
        let chunks = TextChunks::new(16, &Characters, SemanticLevel::Sentence, text, false)
            .map(|(_, s)| s)
            .collect::<Vec<_>>();
        assert_eq!(
            vec!["Mr. Fox jumped. ", "[...] ", "The dog was too ", "lazy."],
            chunks
        );
    }

    #[test]
    fn trim_sentence_indices() {
        let text = "Some text. From a document.";
        let chunks = TextChunks::new(10, &Characters, SemanticLevel::Sentence, text, true)
            .collect::<Vec<_>>();
        assert_eq!(
            vec![(0, "Some text."), (11, "From a"), (18, "document.")],
            chunks
        );
    }

    #[test]
    fn trim_paragraph_indices() {
        let text = "Some text\n\nfrom a\ndocument";
        let chunks = TextChunks::new(10, &Characters, SemanticLevel::DoubleLineBreak, text, true)
            .collect::<Vec<_>>();
        assert_eq!(
            vec![(0, "Some text"), (11, "from a"), (18, "document")],
            chunks
        );
    }
}

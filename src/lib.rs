/*!
# text-splitter

Large language models (LLMs) can be used for many tasks, but often have a limited context size that can be smaller than documents you might want to use. To use documents of larger length, you often have to split your text into chunks to fit within this context size.

This crate provides methods for splitting longer pieces of text into smaller chunks, aiming to maximize a desired chunk size, but still splitting at semantically sensible boundaries whenever possible.

## Get Started

### By Number of Characters

```rust
use text_splitter::{Characters, TextSplitter};

// Maximum number of characters in a chunk
let max_characters = 1000;
let splitter = TextSplitter::new(Characters)
    // Optionally can also have the splitter trim whitespace for you
    .with_trim_chunks(true);

let chunks = splitter.chunks("your document text", max_characters);
```

### By Tokens

```rust
use text_splitter::{TextSplitter};
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
  a. Yes. Merge as many of these neighboring sections into a chunk as possible to maximize chunk length.
  b. No. Split by the next level and repeat.

The boundaries used to split the text if using the top-level `split` method, in descending length:

1. 2 or more newlines (Newline is `\r\n`, `\n`, or `\r`)
2. 1 newline
3. [Unicode Sentences](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
4. [Unicode Words](https://www.unicode.org/reports/tr29/#Word_Boundaries)
5. [Unicode Graphemes](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
6. Characters

Splitting doesn't occur below the character level, otherwise you could get partial bytes of a char, which may not be a valid unicode str.

*Note on sentences:* There are lots of methods of determining sentence breaks, all to varying degrees of accuracy, and many requiring ML models to do so. Rather than trying to find the perfect sentence breaks, we rely on unicode method of sentence boundaries, which in most cases is good enough for finding a decent semantic breaking point if a paragraph is too large, and avoids the performance penalties of many other methods.

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

use core::iter::once;

use either::Either;
use itertools::Itertools;
use once_cell::sync::Lazy;
use regex::Regex;
use unicode_segmentation::UnicodeSegmentation;

mod characters;
#[cfg(feature = "huggingface")]
mod huggingface;
#[cfg(feature = "tiktoken")]
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

/// Default plain-text splitter. Recursively splits chunks into the smallest
/// semantic units that fit within the chunk size. Also will attempt to merge
/// neighboring chunks if they can fit within the given chunk size.
#[derive(Debug)]
pub struct TextSplitter<C>
where
    C: ChunkValidator,
{
    /// Method of determining chunk sizes.
    chunk_validator: C,
    /// Whether or not all chunks should have whitespace trimmed.
    /// If `false`, joining all chunks should return the original string.
    /// If `true`, all chunks will have whitespace removed from beginning and end.
    trim_chunks: bool,
}

// Lazy so that we don't have to compile them more than once
/// Any sequence of 2 or more newlines
static DOUBLE_NEWLINE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\r\n){2,}|\r{2,}|\n{2,}").unwrap());
/// Fallback for anything else
static NEWLINE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\r|\n)+").unwrap());

impl<C> TextSplitter<C>
where
    C: ChunkValidator,
{
    /// Creates a new [`TextSplitter`].
    ///
    /// ```
    /// use text_splitter::{Characters, TextSplitter};
    ///
    /// let splitter = TextSplitter::new(Characters);
    /// ```
    #[must_use]
    pub fn new(chunk_size: C) -> Self {
        Self {
            chunk_validator: chunk_size,
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
    /// let splitter = TextSplitter::new(Characters).with_trim_chunks(true);
    /// ```
    #[must_use]
    pub fn with_trim_chunks(mut self, trim_chunks: bool) -> Self {
        self.trim_chunks = trim_chunks;
        self
    }

    /// Is the given text within the chunk size?
    fn is_within_chunk_size(&self, chunk: &str, chunk_size: usize) -> bool {
        self.chunk_validator.validate_chunk(
            if self.trim_chunks {
                chunk.trim()
            } else {
                chunk
            },
            chunk_size,
        )
    }

    /// Internal method to handle chunk splitting for anything above char level.
    /// Merges neighboring chunks, and also assumes that all chunks in the iterator
    /// are already less than the chunk size.
    ///
    /// Any elements that are above the chunk size limit will be included.
    fn generate_chunks_from_str_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
        chunk_size: usize,
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
                if !self.is_within_chunk_size(chunk, chunk_size) && end != 0 {
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

    /// Returns an iterator over the characters of the text and their byte offsets.
    /// Each chunk will be up to the `max_chunk_size`.
    ///
    /// If a text is too large, each chunk will fit as many `char`s as
    /// possible.
    ///
    /// If you chunk size is smaller than a given character, the character will be returned anyway, otherwise you would get just partial bytes of a char
    /// that might not be a valid unicode str.
    fn chunk_by_char_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
        chunk_size: usize,
    ) -> impl Iterator<Item = (usize, &'b str)> + 'a {
        self.generate_chunks_from_str_indices(
            text,
            chunk_size,
            text.char_indices().map(|(i, c)| {
                (
                    i,
                    text.get(i..i + c.len_utf8()).expect("char should be valid"),
                )
            }),
        )
    }

    /// Returns an iterator over the grapheme clusters of the text and their byte offsets.
    /// Each chunk will be up to the `max_chunk_size`.
    ///
    /// If a text is too large, each chunk will fit as many
    /// [unicode graphemes](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
    /// as possible.
    ///
    /// If a given grapheme is larger than your chunk size, given the length
    /// function, then it will be passed through
    /// [`TextSplitter::chunk_by_char_indices`] until it will fit in a chunk.
    /// ```
    fn chunk_by_grapheme_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
        chunk_size: usize,
    ) -> impl Iterator<Item = (usize, &'b str)> + 'a {
        self.generate_chunks_from_str_indices(
            text,
            chunk_size,
            text.grapheme_indices(true).flat_map(move |(i, grapheme)| {
                if self.is_within_chunk_size(grapheme, chunk_size) {
                    Either::Left(once((i, grapheme)))
                } else {
                    // If grapheme is too large, do char chunking
                    Either::Right(
                        self.chunk_by_char_indices(grapheme, chunk_size)
                            // Offset relative indices back to parent string
                            .map(move |(ci, c)| (ci + i, c)),
                    )
                }
            }),
        )
    }

    /// Returns an iterator over the words of the text and their byte offsets.
    /// Each chunk will be up to the `max_chunk_size`.
    ///
    /// If a text is too large, each chunk will fit as many
    /// [unicode words](https://www.unicode.org/reports/tr29/#Word_Boundaries)
    /// as possible.
    ///
    /// If a given word is larger than your chunk size, given the length
    /// function, then it will be passed through
    /// [`TextSplitter::chunk_by_grapheme_indices`] until it will fit in a chunk.
    fn chunk_by_word_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
        chunk_size: usize,
    ) -> impl Iterator<Item = (usize, &'b str)> + 'a {
        self.generate_chunks_from_str_indices(
            text,
            chunk_size,
            text.split_word_bound_indices().flat_map(move |(i, word)| {
                if self.is_within_chunk_size(word, chunk_size) {
                    Either::Left(once((i, word)))
                } else {
                    // If words is too large, do grapheme chunking
                    Either::Right(
                        self.chunk_by_grapheme_indices(word, chunk_size)
                            // Offset relative indices back to parent string
                            .map(move |(gi, g)| (gi + i, g)),
                    )
                }
            }),
        )
    }

    /// Returns an iterator over the unicode sentences of the text and their byte offsets. Each chunk will be up to
    /// the `max_chunk_size`.
    ///
    /// If a text is too large, each chunk will fit as many
    /// [unicode sentences](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
    /// as possible.
    ///
    /// If a given sentence is larger than your chunk size, given the length
    /// function, then it will be passed through
    /// [`TextSplitter::chunk_by_word_indices`] until it will fit in a chunk.
    fn chunk_by_sentence_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
        chunk_size: usize,
    ) -> impl Iterator<Item = (usize, &'b str)> + 'a {
        self.generate_chunks_from_str_indices(
            text,
            chunk_size,
            text.split_sentence_bound_indices()
                .flat_map(move |(i, sentence)| {
                    if self.is_within_chunk_size(sentence, chunk_size) {
                        Either::Left(once((i, sentence)))
                    } else {
                        // If sentence is too large, do word chunking
                        Either::Right(
                            self.chunk_by_word_indices(sentence, chunk_size)
                                // Offset relative indices back to parent string
                                .map(move |(wi, w)| (wi + i, w)),
                        )
                    }
                }),
        )
    }

    /// Returns an iterator over the paragraphs of the text and their byte offsets.
    /// Each chunk will be up to the `max_chunk_size`.
    ///
    /// If a text is too large, each chunk will fit as many paragraphs as
    /// possible by single newlines.
    ///
    /// If a given paragraph is larger than your chunk size, given the length
    /// function, then it will be passed through
    /// [`TextSplitter::chunk_by_sentence_indices`] until it will fit in a chunk.
    fn chunk_by_newline_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
        chunk_size: usize,
    ) -> impl Iterator<Item = (usize, &'b str)> + 'a {
        self.generate_chunks_from_str_indices(
            text,
            chunk_size,
            Self::str_indices_from_regex_separator(text, &NEWLINE).flat_map(
                move |(i, paragraph)| {
                    if self.is_within_chunk_size(paragraph, chunk_size) {
                        Either::Left(once((i, paragraph)))
                    } else {
                        // If paragraph is still too large, do sentences
                        Either::Right(
                            self.chunk_by_sentence_indices(paragraph, chunk_size)
                                // Offset relative indices back to parent string
                                .map(move |(si, s)| (si + i, s)),
                        )
                    }
                },
            ),
        )
    }

    /// Returns an iterator over the paragraphs of the text and their byte offsets.
    /// Each chunk will be up to the `max_chunk_size`.
    ///
    /// If a text is too large, each chunk will fit as many paragraphs as
    /// possible, splitting by two or more newlines (checking for both \r
    /// and \n)/
    ///
    /// If a given paragraph is larger than your chunk size, given the length
    /// function, then it will be passed through
    /// [`TextSplitter::chunk_by_newline_indices`] until it will fit in a chunk.
    fn chunk_by_double_newline_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
        chunk_size: usize,
    ) -> impl Iterator<Item = (usize, &'b str)> + 'a {
        self.generate_chunks_from_str_indices(
            text,
            chunk_size,
            Self::str_indices_from_regex_separator(text, &DOUBLE_NEWLINE).flat_map(
                move |(i, paragraph)| {
                    if self.is_within_chunk_size(paragraph, chunk_size) {
                        Either::Left(once((i, paragraph)))
                    } else {
                        // If paragraph is still too large, do single newline
                        Either::Right(
                            self.chunk_by_newline_indices(paragraph, chunk_size)
                                // Offset relative indices back to parent string
                                .map(move |(si, s)| (si + i, s)),
                        )
                    }
                },
            ),
        )
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
    /// 3. [Unicode Sentences](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
    /// 4. [Unicode Words](https://www.unicode.org/reports/tr29/#Word_Boundaries)
    /// 5. [Unicode Graphemes](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
    /// 6. Characters
    ///
    /// Splitting doesn't occur below the character level, otherwise you could get partial
    /// bytes of a char, which may not be a valid unicode str.
    ///
    /// ```
    /// use text_splitter::{Characters, TextSplitter};
    ///
    /// let splitter = TextSplitter::new(Characters);
    /// let text = "Some text\n\nfrom a\ndocument";
    /// let chunks = splitter.chunks(text, 10).collect::<Vec<_>>();
    ///
    /// assert_eq!(vec!["Some text", "\n\nfrom a\n", "document"], chunks);
    /// ```
    pub fn chunks<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
        chunk_size: usize,
    ) -> impl Iterator<Item = &'b str> + 'a {
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
    /// let splitter = TextSplitter::new(Characters);
    /// let text = "Some text\n\nfrom a\ndocument";
    /// let chunks = splitter.chunk_indices(text, 10).collect::<Vec<_>>();
    ///
    /// assert_eq!(vec![(0, "Some text"), (9, "\n\nfrom a\n"), (18, "document")], chunks);
    pub fn chunk_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
        chunk_size: usize,
    ) -> impl Iterator<Item = (usize, &'b str)> + 'a {
        self.chunk_by_double_newline_indices(text, chunk_size)
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::min;

    use fake::{Fake, Faker};

    use super::*;

    #[test]
    fn returns_one_chunk_if_text_is_shorter_than_max_chunk_size() {
        let text = Faker.fake::<String>();
        let splitter = TextSplitter::new(Characters);
        let chunks = splitter
            .chunk_by_char_indices(&text, text.chars().count())
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

        let splitter = TextSplitter::new(Characters);
        let chunks = splitter
            .chunk_by_char_indices(&text, max_chunk_size)
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
        let splitter = TextSplitter::new(Characters);
        let chunks = splitter
            .chunk_by_char_indices(text, 100)
            .map(|(_, c)| c)
            .collect::<Vec<_>>();
        assert!(chunks.is_empty());
    }

    #[test]
    fn can_handle_unicode_characters() {
        let text = "éé"; // Char that is more than one byte
        let splitter = TextSplitter::new(Characters);
        let chunks = splitter
            .chunk_by_char_indices(text, 1)
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
        let splitter = TextSplitter::new(Str);
        let chunks = splitter
            .chunk_by_char_indices(text, 2)
            .map(|(_, c)| c)
            .collect::<Vec<_>>();
        assert_eq!(vec!["é", "é"], chunks);
    }

    #[test]
    fn handles_char_bigger_than_len() {
        let text = "éé"; // Char that is two bytes each
        let splitter = TextSplitter::new(Str);
        let chunks = splitter
            .chunk_by_char_indices(text, 1)
            .map(|(_, c)| c)
            .collect::<Vec<_>>();
        // We can only go so small
        assert_eq!(vec!["é", "é"], chunks);
    }

    #[test]
    fn chunk_by_graphemes() {
        let text = "a̐éö̲\r\n";
        let splitter = TextSplitter::new(Characters);

        let chunks = splitter
            .chunk_by_grapheme_indices(text, 3)
            .map(|(_, g)| g)
            .collect::<Vec<_>>();
        // \r\n is grouped together not separated
        assert_eq!(vec!["a̐é", "ö̲", "\r\n"], chunks);
    }

    #[test]
    fn trim_char_indices() {
        let text = " a b ";
        let splitter = TextSplitter::new(Characters).with_trim_chunks(true);

        let chunks = splitter.chunk_by_char_indices(text, 1).collect::<Vec<_>>();
        assert_eq!(vec![(1, "a"), (3, "b")], chunks);
    }

    #[test]
    fn graphemes_fallback_to_chars() {
        let text = "a̐éö̲\r\n";
        let splitter = TextSplitter::new(Characters);

        let chunks = splitter
            .chunk_by_grapheme_indices(text, 1)
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
        let splitter = TextSplitter::new(Characters).with_trim_chunks(true);

        let chunks = splitter
            .chunk_by_grapheme_indices(text, 3)
            .collect::<Vec<_>>();
        assert_eq!(vec![(2, "a̐é"), (7, "ö̲")], chunks);
    }

    #[test]
    fn chunk_by_words() {
        let text = "The quick (\"brown\") fox can't jump 32.3 feet, right?";
        let splitter = TextSplitter::new(Characters);

        let chunks = splitter
            .chunk_by_word_indices(text, 10)
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
        let splitter = TextSplitter::new(Characters);

        let chunks = splitter
            .chunk_by_word_indices(text, 2)
            .map(|(_, w)| w)
            .collect::<Vec<_>>();
        assert_eq!(vec!["Th", "é ", "qu", "ic", "k", "\r\n"], chunks);
    }

    #[test]
    fn trim_word_indices() {
        let text = "Some text from a document";
        let splitter = TextSplitter::new(Characters).with_trim_chunks(true);

        let chunks = splitter.chunk_by_word_indices(text, 10).collect::<Vec<_>>();
        assert_eq!(
            vec![(0, "Some text"), (10, "from a"), (17, "document")],
            chunks
        );
    }

    #[test]
    fn chunk_by_sentences() {
        let text = "Mr. Fox jumped. [...] The dog was too lazy.";
        let splitter = TextSplitter::new(Characters);

        let chunks = splitter
            .chunk_by_sentence_indices(text, 21)
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
        let splitter = TextSplitter::new(Characters);

        let chunks = splitter
            .chunk_by_sentence_indices(text, 16)
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
        let splitter = TextSplitter::new(Characters).with_trim_chunks(true);

        let chunks = splitter
            .chunk_by_sentence_indices(text, 10)
            .collect::<Vec<_>>();
        assert_eq!(
            vec![(0, "Some text."), (11, "From a"), (18, "document.")],
            chunks
        );
    }

    #[test]
    fn trim_paragraph_indices() {
        let text = "Some text\n\nfrom a\ndocument";
        let splitter = TextSplitter::new(Characters).with_trim_chunks(true);

        let chunks = splitter
            .chunk_by_double_newline_indices(text, 10)
            .collect::<Vec<_>>();
        assert_eq!(
            vec![(0, "Some text"), (11, "from a"), (18, "document")],
            chunks
        );
    }
}

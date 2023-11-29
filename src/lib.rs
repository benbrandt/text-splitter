/*!
# text-splitter

[![Docs](https://docs.rs/text-splitter/badge.svg)](https://docs.rs/text-splitter/)
[![Licence](https://img.shields.io/crates/l/text-splitter)](https://github.com/benbrandt/text-splitter/blob/main/LICENSE.txt)
[![Crates.io](https://img.shields.io/crates/v/text-splitter)](https://crates.io/crates/text-splitter)
[![codecov](https://codecov.io/github/benbrandt/text-splitter/branch/main/graph/badge.svg?token=TUF1IAI7G7)](https://codecov.io/github/benbrandt/text-splitter)

- **Rust Crate**: [text-splitter](https://crates.io/crates/text-splitter)
- **Python Bindings**: [semantic-text-splitter](https://pypi.org/project/semantic-text-splitter/) (unfortunately couldn't acquire the same package name)

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

### With Huggingface Tokenizer

Requires the `tokenizers` feature to be activated.

```rust
use text_splitter::TextSplitter;
// Can also use anything else that implements the ChunkSizer
// trait from the text_splitter crate.
use tokenizers::Tokenizer;

let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
let max_tokens = 1000;
let splitter = TextSplitter::new(tokenizer)
    // Optionally can also have the splitter trim whitespace for you
    .with_trim_chunks(true);

let chunks = splitter.chunks("your document text", max_tokens);
```

### With Tiktoken Tokenizer

Requires the `tiktoken-rs` feature to be activated.

```rust
use text_splitter::TextSplitter;
// Can also use anything else that implements the ChunkSizer
// trait from the text_splitter crate.
use tiktoken_rs::cl100k_base;

let tokenizer = cl100k_base().unwrap();
let max_tokens = 1000;
let splitter = TextSplitter::new(tokenizer)
    // Optionally can also have the splitter trim whitespace for you
    .with_trim_chunks(true);

let chunks = splitter.chunks("your document text", max_tokens);
```

### Using a Range for Chunk Capacity

You also have the option of specifying your chunk capacity as a range.

Once a chunk has reached a length that falls within the range it will be returned.

It is always possible that a chunk may be returned that is less than the `start` value, as adding the next piece of text may have made it larger than the `end` capacity.

```rust
use text_splitter::{Characters, TextSplitter};

// Maximum number of characters in a chunk. Will fill up the
// chunk until it is somewhere in this range.
let max_characters = 500..2000;
// Default implementation uses character count for chunk size
let splitter = TextSplitter::default().with_trim_chunks(true);

let chunks = splitter.chunks("your document text", max_characters);
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

use core::{
    cmp::Ordering,
    iter::once,
    ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
};

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

pub use characters::Characters;

/// Determines the size of a given chunk.
pub trait ChunkSizer {
    /// Determine the size of a given chunk to use for validation
    fn chunk_size(&self, chunk: &str) -> usize;
}

/// Describes the largest valid chunk size(s) that can be generated.
///
/// An `end` size is required, which is the maximum possible chunk size that
/// can be generated.
///
/// A `start` size is optional. By specifying `start` and `end` it means a
/// range of sizes will be considered valid. Once a chunk has reached a length
/// that falls between `start` and `end` it will be returned.
///
/// It is always possible that a chunk may be returned that is less than the
/// `start` value, as adding the next piece of text may have made it larger
/// than the `end` capacity.
pub trait ChunkCapacity {
    /// An optional `start` value. If both `start` and `end` are specified, a
    /// valid chunk can fall anywhere between the two values (inclusive).
    fn start(&self) -> Option<usize> {
        None
    }

    /// The maximum size that a chunk can be.
    #[must_use]
    fn end(&self) -> usize;

    /// Validate if a given chunk fits within the capacity
    ///
    /// - `Ordering::Less` indicates more could be added
    /// - `Ordering::Equal` indicates the chunk is within the capacity range
    /// - `Ordering::Greater` indicates the chunk is larger than the capacity
    fn fits(&self, chunk_size: usize) -> Ordering {
        let end = self.end();

        match self.start() {
            Some(start) => {
                if chunk_size < start {
                    Ordering::Less
                } else if chunk_size > end {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            }
            None => chunk_size.cmp(&end),
        }
    }
}

impl ChunkCapacity for usize {
    fn end(&self) -> usize {
        *self
    }
}

impl ChunkCapacity for Range<usize> {
    fn start(&self) -> Option<usize> {
        Some(self.start)
    }

    fn end(&self) -> usize {
        (if self.end > 0 { self.end - 1 } else { 0 }).max(self.start)
    }
}

impl ChunkCapacity for RangeFrom<usize> {
    fn start(&self) -> Option<usize> {
        Some(self.start)
    }

    fn end(&self) -> usize {
        usize::MAX
    }
}

impl ChunkCapacity for RangeFull {
    fn start(&self) -> Option<usize> {
        Some(usize::MIN)
    }

    fn end(&self) -> usize {
        usize::MAX
    }
}

impl ChunkCapacity for RangeInclusive<usize> {
    fn start(&self) -> Option<usize> {
        Some(*self.start())
    }

    fn end(&self) -> usize {
        *self.end()
    }
}

impl ChunkCapacity for RangeTo<usize> {
    fn start(&self) -> Option<usize> {
        Some(usize::MIN)
    }

    fn end(&self) -> usize {
        if self.end > 0 {
            self.end - 1
        } else {
            0
        }
    }
}

impl ChunkCapacity for RangeToInclusive<usize> {
    fn start(&self) -> Option<usize> {
        Some(usize::MIN)
    }

    fn end(&self) -> usize {
        self.end
    }
}

/// Default plain-text splitter. Recursively splits chunks into the largest
/// semantic units that fit within the chunk size. Also will attempt to merge
/// neighboring chunks if they can fit within the given chunk size.
#[derive(Debug)]
pub struct TextSplitter<S>
where
    S: ChunkSizer,
{
    /// Method of determining chunk sizes.
    chunk_sizer: S,
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

impl<S> TextSplitter<S>
where
    S: ChunkSizer,
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
    pub fn new(chunk_sizer: S) -> Self {
        Self {
            chunk_sizer,
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
        chunk_capacity: impl ChunkCapacity + 'splitter,
    ) -> impl Iterator<Item = &'text str> + 'splitter {
        self.chunk_indices(text, chunk_capacity).map(|(_, t)| t)
    }

    /// Returns an iterator over chunks of the text and their byte offsets.
    /// Each chunk will be up to the `chunk_capacity`.
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
        chunk_capacity: impl ChunkCapacity + 'splitter,
    ) -> impl Iterator<Item = (usize, &'text str)> + 'splitter {
        TextChunks::new(chunk_capacity, &self.chunk_sizer, text, self.trim_chunks)
    }
}

/// Different semantic levels that text can be split by.
/// Each level provides a method of splitting text into chunks of a given level
/// as well as a fallback in case a given fallback is too large.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
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
    /// Split by given number of linebreaks, either `\n`, `\r`, or `\r\n`.
    /// Falls back to the next lower number, or else [`Self::Sentence`]
    LineBreak(usize),
}

// Lazy so that we don't have to compile them more than once
static LINEBREAKS: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\r\n)+|\r+|\n+").unwrap());

/// Captures information about linebreaks for a given text, and their
/// various semantic levels.
#[derive(Debug)]
struct LineBreaks {
    /// Range of each line break and its precalculated semantic level
    line_breaks: Vec<(SemanticLevel, Range<usize>)>,
    /// Maximum number of linebreaks in a given text
    max_level: SemanticLevel,
}

impl LineBreaks {
    /// Generate linebreaks for a given text
    fn new(text: &str) -> Self {
        let linebreaks = LINEBREAKS
            .find_iter(text)
            .map(|m| {
                let range = m.range();
                let level = text
                    .get(range.start..range.end)
                    .unwrap()
                    .graphemes(true)
                    .count();
                (
                    match level {
                        0 => SemanticLevel::Sentence,
                        n => SemanticLevel::LineBreak(n),
                    },
                    range,
                )
            })
            .collect::<Vec<_>>();

        let max_level = *linebreaks
            .iter()
            .map(|(l, _)| l)
            .max_by_key(|level| match level {
                SemanticLevel::LineBreak(n) => n,
                _ => &0,
            })
            .unwrap_or(&SemanticLevel::Sentence);

        Self {
            line_breaks: linebreaks,
            max_level,
        }
    }

    /// Retrieve ranges for all linebreaks of a given level after an offset
    fn ranges(
        &self,
        offset: usize,
        level: SemanticLevel,
    ) -> impl Iterator<Item = &(SemanticLevel, Range<usize>)> + '_ {
        self.line_breaks
            .iter()
            .filter(move |(l, sep)| l >= &level && sep.start >= offset)
    }

    /// Return a unique, sorted list of all line break levels present before the next max level, added
    /// to all of the base semantic levels, in order from smallest to largest
    fn levels_in_next_max_chunk(&self, offset: usize) -> impl Iterator<Item = SemanticLevel> + '_ {
        let line_break_levels = self
            .line_breaks
            .iter()
            // Only start taking them from the offset
            .filter(|(_, sep)| sep.start >= offset)
            // Stop once we hit the first of the max level
            .take_while(|(l, _)| l < &self.max_level)
            .map(|(l, _)| l)
            .copied();

        [
            SemanticLevel::Char,
            SemanticLevel::GraphemeCluster,
            SemanticLevel::Word,
            SemanticLevel::Sentence,
            self.max_level,
        ]
        .into_iter()
        .chain(line_break_levels)
        .sorted()
        .dedup()
    }
}

/// Returns chunks of text with their byte offsets as an iterator.
#[derive(Debug)]
struct TextChunks<'text, 'sizer, C, S>
where
    C: ChunkCapacity,
    S: ChunkSizer,
{
    /// Size of the chunks to generate
    chunk_capacity: C,
    /// How to validate chunk sizes
    chunk_sizer: &'sizer S,
    /// Current byte offset in the `text`
    cursor: usize,
    /// Ranges where linebreaks occur. Save to optimize how many regex
    /// passes we need to do.
    line_breaks: LineBreaks,
    /// Original text to iterate over and generate chunks from
    text: &'text str,
    /// Whether or not chunks should be trimmed
    trim_chunks: bool,
}

impl<'text, 'sizer, C, S> TextChunks<'text, 'sizer, C, S>
where
    C: ChunkCapacity,
    S: ChunkSizer,
{
    /// Generate new [`TextChunks`] iterator for a given text.
    /// Starts with an offset of 0
    fn new(chunk_capacity: C, chunk_sizer: &'sizer S, text: &'text str, trim_chunks: bool) -> Self {
        Self {
            cursor: 0,
            chunk_capacity,
            chunk_sizer,
            line_breaks: LineBreaks::new(text),
            text,
            trim_chunks,
        }
    }

    /// Is the given text within the chunk size?
    fn check_capacity(&self, chunk: &str) -> (usize, Ordering) {
        let chunk_size = self.chunk_sizer.chunk_size(if self.trim_chunks {
            chunk.trim()
        } else {
            chunk
        });
        (chunk_size, self.chunk_capacity.fits(chunk_size))
    }

    /// Generate the next chunk, applying trimming settings.
    /// Returns final byte offset and str.
    /// Will return `None` if given an invalid range.
    fn next_chunk(&mut self) -> Option<(usize, &'text str)> {
        let start = self.cursor;
        let mut end_offset = self.cursor;
        let section_lengths: Vec<usize> = self.next_sections()?.map(|(_, s)| s.len()).collect();

        // Search for the largest chunk that satisfies the capacity requirements
        let mut low = 0;
        let mut high = section_lengths.len() - 1;
        let mut found_offset = end_offset;

        while low <= high {
            let mid = low + (high - low) / 2;
            end_offset = section_lengths.iter().take(mid + 1).sum::<usize>() + start;
            let candidate = self.text.get(start..end_offset).expect("Tried to seek to invalid offset");
            let (_, fits) = self.check_capacity(candidate);

            // If we're too big on our smallest run, we must return at least one section
            if mid == 0 && fits.is_gt() {
                found_offset = end_offset;
                break;
            }

            if fits.is_eq() || fits.is_lt() {
                low = mid + 1;
                found_offset = end_offset;
            } else {
                high = mid - 1;
            }
        }

        self.cursor = found_offset;
        // Empty input will return None from here
        let chunk = self.text.get(start..self.cursor)?;

        // Trim whitespace if user requested it
        Some(if self.trim_chunks {
            // Figure out how many bytes we lose trimming the beginning
            let offset = chunk.len() - chunk.trim_start().len();
            (start + offset, chunk.trim())
        } else {
            (start, chunk)
        })
    }

    /// Split a given text into iterator over each semantic chunk
    #[auto_enum(Iterator)]
    fn semantic_chunks(
        &self,
        semantic_level: SemanticLevel,
    ) -> impl Iterator<Item = (usize, &'text str)> + '_ {
        let text = self.text.get(self.cursor..).unwrap();
        match semantic_level {
            SemanticLevel::Char => text.char_indices().map(|(i, c)| {
                (
                    self.cursor + i,
                    text.get(i..i + c.len_utf8()).expect("char should be valid"),
                )
            }),
            SemanticLevel::GraphemeCluster => text
                .grapheme_indices(true)
                .map(|(i, str)| (self.cursor + i, str)),
            SemanticLevel::Word => text
                .split_word_bound_indices()
                .map(|(i, str)| (self.cursor + i, str)),
            SemanticLevel::Sentence => text
                .split_sentence_bound_indices()
                .map(|(i, str)| (self.cursor + i, str)),
            SemanticLevel::LineBreak(_) => split_str_by_separator(
                text,
                true,
                self.line_breaks
                    .ranges(self.cursor, semantic_level)
                    .map(|(_, sep)| sep.start - self.cursor..sep.end - self.cursor),
            )
            .map(|(i, str)| (self.cursor + i, str)),
        }
    }

    /// Find the ideal next sections, breaking it up until we find the largest chunk.
    /// Increasing length of chunk until we find biggest size to minimize validation time
    /// on huge chunks
    fn next_sections(&self) -> Option<impl Iterator<Item = (usize, &'text str)> + '_> {
        // Next levels to try. Will stop at max level. We check only levels in the next max level
        // chunk so we don't bypass it if not all levels are present in every chunk.
        let mut levels = self.line_breaks.levels_in_next_max_chunk(self.cursor);
        // Get starting level
        let mut semantic_level = levels.next()?;
        // If we aren't at the highest semantic level, stop iterating sections that go beyond the range of the next level.
        let mut max_offset = None;

        for level in levels {
            let (offset, str) = self.semantic_chunks(level).next()?;
            // If this no longer fits, we use the level we are at. Or if we already
            // have the rest of the string
            let (_, fits) = self.check_capacity(str);
            if fits.is_gt() || self.text.get(self.cursor..)? == str {
                max_offset = Some(offset + str.len());
                break;
            }
            // Otherwise break up the text with the next level
            semantic_level = level;
        }

        Some(
            self.semantic_chunks(semantic_level)
                // We don't want to return items at this level that go beyond the next highest semantic level, as that is most
                // likely a meaningful breakpoint we want to preserve. We already know that the next highest doesn't fit anyway,
                // so we should be safe to break once we reach it.
                .take_while(move |(offset, _)| max_offset.map_or(true, |max| offset < &max)),
        )
    }
}

impl<'text, 'sizer, C, S> Iterator for TextChunks<'text, 'sizer, C, S>
where
    C: ChunkCapacity,
    S: ChunkSizer,
{
    type Item = (usize, &'text str);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Make sure we haven't reached the end
            if self.cursor >= self.text.len() {
                return None;
            }

            match self.next_chunk()? {
                // Make sure we didn't get an empty chunk. Should only happen in
                // cases where we trim.
                (_, chunk) if chunk.is_empty() => continue,
                c => return Some(c),
            }
        }
    }
}

/// Given a list of separator ranges, construct the sections of the text
fn split_str_by_separator(
    text: &str,
    separator_is_own_chunk: bool,
    separator_ranges: impl Iterator<Item = Range<usize>>,
) -> impl Iterator<Item = (usize, &str)> {
    let mut cursor = 0;
    let mut final_match = false;
    separator_ranges
        .batching(move |it| match it.next() {
            // If we've hit the end, actually return None
            None if final_match => None,
            // First time we hit None, return the final section of the text
            None => {
                final_match = true;
                text.get(cursor..).map(|t| Either::Left(once((cursor, t))))
            }
            // Return text preceding match + the match
            Some(range) if separator_is_own_chunk => {
                let offset = cursor;
                let prev_section = text
                    .get(cursor..range.start)
                    .expect("invalid character sequence");
                let separator = text
                    .get(range.start..range.end)
                    .expect("invalid character sequence");
                cursor = range.end;
                Some(Either::Right(
                    [(offset, prev_section), (range.start, separator)].into_iter(),
                ))
            }
            // Return just the text preceding the match
            Some(range) => {
                let offset = cursor;
                let prev_section = text
                    .get(cursor..range.start)
                    .expect("invalid character sequence");
                // Separator will be part of the next chunk
                cursor = range.start;
                Some(Either::Left(once((offset, prev_section))))
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
        let chunks = TextChunks::new(text.chars().count(), &Characters, &text, false)
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

        let chunks = TextChunks::new(max_chunk_size, &Characters, &text, false)
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
        let chunks = TextChunks::new(100, &Characters, text, false)
            .map(|(_, c)| c)
            .collect::<Vec<_>>();
        assert!(chunks.is_empty());
    }

    #[test]
    fn can_handle_unicode_characters() {
        let text = "éé"; // Char that is more than one byte
        let chunks = TextChunks::new(1, &Characters, text, false)
            .map(|(_, c)| c)
            .collect::<Vec<_>>();
        assert_eq!(vec!["é", "é"], chunks);
    }

    // Just for testing
    struct Str;

    impl ChunkSizer for Str {
        fn chunk_size(&self, chunk: &str) -> usize {
            chunk.len()
        }
    }

    #[test]
    fn custom_len_function() {
        let text = "éé"; // Char that is two bytes each
        let chunks = TextChunks::new(2, &Str, text, false)
            .map(|(_, c)| c)
            .collect::<Vec<_>>();
        assert_eq!(vec!["é", "é"], chunks);
    }

    #[test]
    fn handles_char_bigger_than_len() {
        let text = "éé"; // Char that is two bytes each
        let chunks = TextChunks::new(1, &Str, text, false)
            .map(|(_, c)| c)
            .collect::<Vec<_>>();
        // We can only go so small
        assert_eq!(vec!["é", "é"], chunks);
    }

    #[test]
    fn chunk_by_graphemes() {
        let text = "a̐éö̲\r\n";

        let chunks = TextChunks::new(3, &Characters, text, false)
            .map(|(_, g)| g)
            .collect::<Vec<_>>();
        // \r\n is grouped together not separated
        assert_eq!(vec!["a̐é", "ö̲", "\r\n"], chunks);
    }

    #[test]
    fn trim_char_indices() {
        let text = " a b ";

        let chunks = TextChunks::new(1, &Characters, text, true).collect::<Vec<_>>();
        assert_eq!(vec![(1, "a"), (3, "b")], chunks);
    }

    #[test]
    fn graphemes_fallback_to_chars() {
        let text = "a̐éö̲\r\n";

        let chunks = TextChunks::new(1, &Characters, text, false)
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

        let chunks = TextChunks::new(3, &Characters, text, true).collect::<Vec<_>>();
        assert_eq!(vec![(2, "a̐é"), (7, "ö̲")], chunks);
    }

    #[test]
    fn chunk_by_words() {
        let text = "The quick (\"brown\") fox can't jump 32.3 feet, right?";

        let chunks = TextChunks::new(10, &Characters, text, false)
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
        let chunks = TextChunks::new(2, &Characters, text, false)
            .map(|(_, w)| w)
            .collect::<Vec<_>>();
        assert_eq!(vec!["Th", "é ", "qu", "ic", "k", "\r\n"], chunks);
    }

    #[test]
    fn trim_word_indices() {
        let text = "Some text from a document";
        let chunks = TextChunks::new(10, &Characters, text, true).collect::<Vec<_>>();
        assert_eq!(
            vec![(0, "Some text"), (10, "from a"), (17, "document")],
            chunks
        );
    }

    #[test]
    fn chunk_by_sentences() {
        let text = "Mr. Fox jumped. [...] The dog was too lazy.";
        let chunks = TextChunks::new(21, &Characters, text, false)
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
        let chunks = TextChunks::new(16, &Characters, text, false)
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
        let chunks = TextChunks::new(10, &Characters, text, true).collect::<Vec<_>>();
        assert_eq!(
            vec![(0, "Some text."), (11, "From a"), (18, "document.")],
            chunks
        );
    }

    #[test]
    fn trim_paragraph_indices() {
        let text = "Some text\n\nfrom a\ndocument";
        let chunks = TextChunks::new(10, &Characters, text, true).collect::<Vec<_>>();
        assert_eq!(
            vec![(0, "Some text"), (11, "from a"), (18, "document")],
            chunks
        );
    }

    #[test]
    fn correctly_determines_newlines() {
        let text = "\r\n\r\ntext\n\n\ntext2";
        let linebreaks = LineBreaks::new(text);
        assert_eq!(
            vec![
                (SemanticLevel::LineBreak(2), 0..4),
                (SemanticLevel::LineBreak(3), 8..11)
            ],
            linebreaks.line_breaks
        );
        assert_eq!(SemanticLevel::LineBreak(3), linebreaks.max_level);
    }

    #[test]
    fn check_chunk_capacity() {
        let chunk = "12345";

        assert_eq!(4.fits(Characters.chunk_size(chunk)), Ordering::Greater);
        assert_eq!(5.fits(Characters.chunk_size(chunk)), Ordering::Equal);
        assert_eq!(6.fits(Characters.chunk_size(chunk)), Ordering::Less);
    }

    #[test]
    fn check_chunk_capacity_for_range() {
        let chunk = "12345";

        assert_eq!((0..0).fits(Characters.chunk_size(chunk)), Ordering::Greater);
        assert_eq!((0..5).fits(Characters.chunk_size(chunk)), Ordering::Greater);
        assert_eq!((5..6).fits(Characters.chunk_size(chunk)), Ordering::Equal);
        assert_eq!((6..100).fits(Characters.chunk_size(chunk)), Ordering::Less);
    }

    #[test]
    fn check_chunk_capacity_for_range_from() {
        let chunk = "12345";

        assert_eq!((0..).fits(Characters.chunk_size(chunk)), Ordering::Equal);
        assert_eq!((5..).fits(Characters.chunk_size(chunk)), Ordering::Equal);
        assert_eq!((6..).fits(Characters.chunk_size(chunk)), Ordering::Less);
    }

    #[test]
    fn check_chunk_capacity_for_range_full() {
        let chunk = "12345";

        assert_eq!((..).fits(Characters.chunk_size(chunk)), Ordering::Equal);
    }

    #[test]
    fn check_chunk_capacity_for_range_inclusive() {
        let chunk = "12345";

        assert_eq!(
            (0..=4).fits(Characters.chunk_size(chunk)),
            Ordering::Greater
        );
        assert_eq!((5..=6).fits(Characters.chunk_size(chunk)), Ordering::Equal);
        assert_eq!((4..=5).fits(Characters.chunk_size(chunk)), Ordering::Equal);
        assert_eq!((6..=100).fits(Characters.chunk_size(chunk)), Ordering::Less);
    }

    #[test]
    fn check_chunk_capacity_for_range_to() {
        let chunk = "12345";

        assert_eq!((..0).fits(Characters.chunk_size(chunk)), Ordering::Greater);
        assert_eq!((..5).fits(Characters.chunk_size(chunk)), Ordering::Greater);
        assert_eq!((..6).fits(Characters.chunk_size(chunk)), Ordering::Equal);
    }

    #[test]
    fn check_chunk_capacity_for_range_to_inclusive() {
        let chunk = "12345";

        assert_eq!((..=4).fits(Characters.chunk_size(chunk)), Ordering::Greater);
        assert_eq!((..=5).fits(Characters.chunk_size(chunk)), Ordering::Equal);
        assert_eq!((..=6).fits(Characters.chunk_size(chunk)), Ordering::Equal);
    }
}

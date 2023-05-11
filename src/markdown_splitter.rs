/*!
# [`MarkdownSplitter`]

Semantic splitting of Markdown documents. Tries to use as many semantic units from Markdown
as possible, eventually falling back to the normal [`TextSplitter`] method.
*/

use std::{iter::once, ops::Range};

use either::Either;
use pulldown_cmark::{Event, Parser};

use crate::{str_indices_from_separator, Characters, ChunkValidator, TextSplitter};

/// Markdown splitter. Recursively splits chunks into the largest
/// semantic units that fit within the chunk size. Also will
/// attempt to merge neighboring chunks if they can fit within the
/// given chunk size.
#[derive(Debug)]
pub struct MarkdownSplitter<C>
where
    C: ChunkValidator,
{
    text_splitter: TextSplitter<C>,
}

impl Default for MarkdownSplitter<Characters> {
    fn default() -> Self {
        Self {
            text_splitter: TextSplitter::default(),
        }
    }
}

impl<C> MarkdownSplitter<C>
where
    C: ChunkValidator,
{
    /// Creates a new [`MarkdownSplitter`].
    ///
    /// ```
    /// use text_splitter::{Characters, MarkdownSplitter};
    ///
    /// // Characters is the default, so you can also do `TextSplitter::default()`
    /// let splitter = MarkdownSplitter::new(Characters);
    /// ```
    #[must_use]
    pub fn new(chunk_validator: C) -> Self {
        Self {
            text_splitter: TextSplitter::new(chunk_validator),
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
    /// use text_splitter::{Characters, MarkdownSplitter};
    ///
    /// let splitter = MarkdownSplitter::default().with_trim_chunks(true);
    /// ```
    #[must_use]
    pub fn with_trim_chunks(mut self, trim_chunks: bool) -> Self {
        self.text_splitter = self.text_splitter.with_trim_chunks(trim_chunks);
        self
    }

    fn chunk_by_horizontal_rule<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
        chunk_size: usize,
    ) -> impl Iterator<Item = (usize, &'b str)> + 'a {
        self.text_splitter.coalesce_str_indices(
            text,
            chunk_size,
            str_indices_from_parser_separator(text, |(event, _)| matches!(event, Event::Rule))
                .flat_map(move |(i, str)| {
                    if self.text_splitter.is_within_chunk_size(str, chunk_size) {
                        Either::Left(once((i, str)))
                    } else {
                        // If section is too large, fallback
                        Either::Right(
                            self.text_splitter
                                .chunk_indices(str, chunk_size)
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
    /// ## Method
    ///
    /// To preserve as much semantic meaning within a chunk as possible, a
    /// recursive approach is used, starting at larger semantic units and, if that is
    /// too large, breaking it up into the next largest unit. Here is an example of the
    /// steps used:
    ///
    /// 1. Split the text by a given level
    /// 2. For each section, does it fit within the chunk size?
    ///   a. Yes. Merge as many of these neighboring sections into a chunk as possible to maximize chunk length.
    ///   b. No. Split by the next level and repeat.
    ///
    /// The boundaries used to split the text if using the top-level `chunks` method, in descending length:
    ///
    /// 1. 2 or more newlines (Newline is `\r\n`, `\n`, or `\r`)
    /// 2. 1 newline
    /// 3. [Unicode Sentence Boundaries](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
    /// 4. [Unicode Word Boundaries](https://www.unicode.org/reports/tr29/#Word_Boundaries)
    /// 5. [Unicode Graphemes Cluster Boundaries](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
    /// 6. Characters
    ///
    /// Splitting doesn't occur below the character level, otherwise you could get partial
    /// bytes of a char, which may not be a valid unicode str.
    ///
    /// ```
    /// use text_splitter::{Characters, MarkdownSplitter};
    ///
    /// let splitter = MarkdownSplitter::default();
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
    /// See [`MarkdownSplitter::chunks`] for more information.
    ///
    /// ```
    /// use text_splitter::{Characters, MarkdownSplitter};
    ///
    /// let splitter = MarkdownSplitter::default();
    /// let text = "Some text\n\nfrom a\ndocument";
    /// let chunks = splitter.chunk_indices(text, 10).collect::<Vec<_>>();
    ///
    /// assert_eq!(vec![(0, "Some text"), (9, "\n\nfrom a\n"), (18, "document")], chunks);
    pub fn chunk_indices<'a, 'b: 'a>(
        &'a self,
        text: &'b str,
        chunk_size: usize,
    ) -> impl Iterator<Item = (usize, &'b str)> + 'a {
        self.chunk_by_horizontal_rule(text, chunk_size)
    }
}

fn str_indices_from_parser_separator<'a, F>(
    text: &'a str,
    filter: F,
) -> impl Iterator<Item = (usize, &'a str)> + '_
where
    F: for<'f> FnMut(&'f (Event<'_>, Range<usize>)) -> bool + 'a,
{
    str_indices_from_separator(
        text,
        Parser::new(text)
            .into_offset_iter()
            .filter(filter)
            .map(|(_, range)| range),
    )
}

/*!
# [`MarkdownSplitter`]
Semantic splitting of Markdown documents. Tries to use as many semantic units from Markdown
as possible, eventually falling back to the normal [`TextSplitter`] method.
*/

use crate::{Characters, ChunkSizer, TextSplitter};

/// Markdown splitter. Recursively splits chunks into the largest
/// semantic units that fit within the chunk size. Also will
/// attempt to merge neighboring chunks if they can fit within the
/// given chunk size.
#[derive(Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct MarkdownSplitter<S>
where
    S: ChunkSizer,
{
    /// Base text splitter to use
    text_splitter: TextSplitter<S>,
}

impl Default for MarkdownSplitter<Characters> {
    fn default() -> Self {
        Self {
            text_splitter: TextSplitter::default(),
        }
    }
}

impl<S> MarkdownSplitter<S>
where
    S: ChunkSizer,
{
    /// Creates a new [`MarkdownSplitter`].
    ///
    /// ```
    /// use text_splitter::{Characters, MarkdownSplitter};
    ///
    /// // Characters is the default, so you can also do `MarkdownSplitter::default()`
    /// let splitter = MarkdownSplitter::new(Characters);
    /// ```
    #[must_use]
    pub fn new(chunk_sizer: S) -> Self {
        Self {
            text_splitter: TextSplitter::new(chunk_sizer),
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
    /// 1. [Headings](https://spec.commonmark.org/0.30/#atx-headings) - in descending levels
    /// 2. [Thematic Breaks](https://spec.commonmark.org/0.30/#thematic-break)
    /// 3. Progress through the `TextSplitter::chunks` method.
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
    pub fn chunk_indices<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
        chunk_size: usize,
    ) -> impl Iterator<Item = (usize, &'text str)> + 'splitter {
        self.text_splitter.chunk_indices(text, chunk_size)
    }
}

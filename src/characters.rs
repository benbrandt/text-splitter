use crate::ChunkSize;

/// Used for splitting a piece of text into chunks based on the number of
/// characters in each chunk.
#[derive(Debug)]
pub struct Characters {
    /// Maximum size of a chunk, measured in characters.
    max_characters: usize,
}

impl Characters {
    /// Creates a new [`Characters`]. Chunks will be generated based on the
    /// number of characters in the chunk.
    ///
    /// `max_characters` determins what the largest chunk will be.
    ///
    /// ```
    /// use text_splitter::{Characters, TextSplitter};
    ///
    /// let splitter = TextSplitter::new(Characters::new(100));
    /// ```
    #[must_use]
    pub fn new(max_characters: usize) -> Self {
        Self { max_characters }
    }
}

impl ChunkSize for Characters {
    /// Determine if the given chunk still fits within the specified max chunk
    /// size, based on characters.
    ///
    /// ```
    /// use text_splitter::{Characters, ChunkSize};
    ///
    /// let characters = Characters::new(10);
    /// assert!(characters.valid_chunk("hello"));
    /// ```
    fn valid_chunk(&self, chunk: &str) -> bool {
        chunk.chars().count() <= self.max_characters
    }
}

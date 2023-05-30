use crate::ChunkSize;

/// Used for splitting a piece of text into chunks based on the number of
/// characters in each chunk.
///
/// ```
/// use text_splitter::{Characters, TextSplitter};
///
/// let splitter = TextSplitter::new(Characters);
/// ```
#[derive(Debug)]
pub struct Characters;

impl ChunkSize for Characters {
    /// Determine the size of a given chunk to use for validation.
    ///
    /// ```
    /// use text_splitter::{Characters, ChunkSize};
    ///
    /// assert_eq!(Characters.chunk_size("hello"), 5);
    fn chunk_size(&self, chunk: &str) -> usize {
        chunk.chars().count()
    }
}

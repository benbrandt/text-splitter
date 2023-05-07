use crate::ChunkValidator;

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

impl ChunkValidator for Characters {
    /// Determine if the given chunk still fits within the specified max chunk
    /// size, based on characters.
    ///
    /// ```
    /// use text_splitter::{Characters, ChunkValidator};
    ///
    /// assert!(Characters.validate_chunk("hello", 10));
    /// ```
    fn validate_chunk(&self, chunk: &str, chunk_size: usize) -> bool {
        chunk.chars().count() <= chunk_size
    }
}

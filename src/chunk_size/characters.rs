use crate::{ChunkCapacity, ChunkSize, ChunkSizer};

/// Used for splitting a piece of text into chunks based on the number of
/// characters in each chunk.
///
/// ```
/// use text_splitter::TextSplitter;
///
/// // Uses character splitter by default.
/// let splitter = TextSplitter::new(10);
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Characters;

impl ChunkSizer for Characters {
    /// Determine the size of a given chunk to use for validation.
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        ChunkSize::from_size(chunk.chars().count(), capacity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_size() {
        let capacity = 10;
        let offsets = Characters.chunk_size("eé", &capacity.into());
        assert_eq!(offsets, ChunkSize::from_size(2, &capacity.into()));
    }
}

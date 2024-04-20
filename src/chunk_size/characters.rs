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
#[derive(Debug, PartialEq)]
pub struct Characters;

impl ChunkSizer for Characters {
    /// Determine the size of a given chunk to use for validation.
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        let offsets = chunk.char_indices().map(|(i, c)| i..(i + c.len_utf8()));
        ChunkSize::from_offsets(offsets, capacity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_offsets() {
        let capacity = 10;
        let offsets = Characters.chunk_size("eé", &capacity.into());
        assert_eq!(
            offsets,
            ChunkSize::from_offsets([0..1, 1..3].into_iter(), &capacity.into())
        );
    }
}

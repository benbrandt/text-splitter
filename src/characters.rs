use std::ops::Range;

use crate::{ChunkCapacity, ChunkSize, ChunkSizer};

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

impl Characters {
    fn encoded_offsets(chunk: &str) -> impl Iterator<Item = Range<usize>> + '_ {
        chunk.char_indices().map(|(i, c)| i..(i + c.len_utf8()))
    }
}

impl ChunkSizer for Characters {
    /// Determine the size of a given chunk to use for validation.
    fn chunk_size(&self, chunk: &str, capacity: &impl ChunkCapacity) -> ChunkSize {
        ChunkSize::from_offsets(Self::encoded_offsets(chunk), capacity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_offsets() {
        let offsets = Characters::encoded_offsets("eé").collect::<Vec<_>>();
        assert_eq!(offsets, vec![0..1, 1..3]);
    }
}

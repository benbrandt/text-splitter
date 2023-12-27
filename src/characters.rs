use crate::{ChunkSizer, EncodedOffsets};

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

impl ChunkSizer for Characters {
    /// Return offsets for each unit of text used to calculate chunk size.
    /// Should return an exclusive byte range for each element counted.
    fn encoded_offsets(&self, chunk: &str) -> EncodedOffsets {
        chunk
            .char_indices()
            .map(|(i, c)| i..(i + c.len_utf8()))
            .collect::<Vec<_>>()
            .into()
    }

    /// Determine the size of a given chunk to use for validation.
    ///
    /// ```
    /// use text_splitter::{Characters, ChunkSizer};
    ///
    /// assert_eq!(Characters.chunk_size("hello"), 5);
    fn chunk_size(&self, chunk: &str) -> usize {
        chunk.chars().count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_offsets() {
        let offsets = Characters.encoded_offsets("eé");
        assert_eq!(offsets, vec![0..1, 1..3].into());
    }
}

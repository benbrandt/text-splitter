use crate::ChunkSizer;

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
    fn size(&self, chunk: &str) -> usize {
        chunk.chars().count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_size() {
        let offsets = Characters.size("eé");
        assert_eq!(offsets, 2);
    }
}

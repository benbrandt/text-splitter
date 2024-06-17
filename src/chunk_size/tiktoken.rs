use tiktoken_rs::CoreBPE;

use crate::{ChunkCapacity, ChunkSize, ChunkSizer};

impl ChunkSizer for &CoreBPE {
    /// Returns the number of tokens in a given text after tokenization.
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        ChunkSize::from_size(self.encode_ordinary(chunk).len(), capacity)
    }
}

impl ChunkSizer for CoreBPE {
    /// Returns the number of tokens in a given text after tokenization.
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        (&self).chunk_size(chunk, capacity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tiktoken_rs::cl100k_base;

    #[test]
    fn returns_offsets() {
        let tokenizer = cl100k_base().unwrap();
        let capacity = 10;
        let offsets = tokenizer.chunk_size("An apple a", &capacity.into());
        assert_eq!(offsets, ChunkSize::from_size(3, &capacity.into()));
    }
}

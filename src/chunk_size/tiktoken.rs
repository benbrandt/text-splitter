use tiktoken_rs::CoreBPE;

use crate::{ChunkCapacity, ChunkSize, ChunkSizer};

impl ChunkSizer for &CoreBPE {
    /// Returns the number of tokens in a given text after tokenization.
    fn chunk_size(&self, chunk: &str, capacity: &impl ChunkCapacity) -> ChunkSize {
        let tokens = self.encode_ordinary(chunk);
        let offsets = self
            ._decode_native_and_split(tokens)
            .scan(0usize, |offset, bytes| {
                let end = *offset + bytes.len();
                let item = *offset..end;
                *offset = end;
                Some(item)
            });
        ChunkSize::from_offsets(offsets, capacity)
    }
}

impl ChunkSizer for CoreBPE {
    /// Returns the number of tokens in a given text after tokenization.
    fn chunk_size(&self, chunk: &str, capacity: &impl ChunkCapacity) -> ChunkSize {
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
        let offsets = tokenizer.chunk_size("An apple a", &capacity);
        assert_eq!(
            offsets,
            ChunkSize::from_offsets([0..2, 2..8, 8..10].into_iter(), &capacity)
        );
    }
}

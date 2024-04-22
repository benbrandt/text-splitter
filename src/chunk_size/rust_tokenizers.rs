use rust_tokenizers::{
    tokenizer::{BertTokenizer, Tokenizer},
    Offset,
};
use std::ops::Range;

use crate::{ChunkCapacity, ChunkSize, ChunkSizer};

impl ChunkSizer for &BertTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        let binding = self.tokenize_with_offsets(chunk);
        let offsets = binding
            .offsets
            .iter()
            .flatten()
            .map(|Offset { begin, end }| Range {
                start: *begin as usize,
                end: *end as usize,
            });
        ChunkSize::from_offsets(offsets, capacity)
    }
}

impl ChunkSizer for BertTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        (&self).chunk_size(chunk, capacity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_offsets() {
        let path: &str = "tests/tokenizers/bert-uncased-vocab.txt";
        let tokenizer: BertTokenizer = BertTokenizer::from_file(path, false, false).unwrap();
        let capacity = 10;
        let offsets = tokenizer.chunk_size(" An apple a", &capacity.into());
        assert_eq!(
            offsets,
            ChunkSize::from_offsets([1..3, 4..9, 10..11].into_iter(), &capacity.into())
        );
    }
}

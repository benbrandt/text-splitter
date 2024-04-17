use rust_tokenizers::{
    tokenizer::{BertTokenizer, Tokenizer, TruncationStrategy},
    Offset,
};
use std::ops::Range;

use crate::{ChunkCapacity, ChunkSize, ChunkSizer};

impl ChunkSizer for &BertTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &impl ChunkCapacity) -> ChunkSize {
        ChunkSize::from_offsets(
            self.encode(
                chunk,
                None,
                capacity.end(),
                &TruncationStrategy::LongestFirst,
                capacity.start().unwrap_or(0),
            )
            .token_offsets
            .iter()
            .flatten()
            .map(|Offset { begin, end }| Range {
                start: *begin as usize,
                end: *end as usize,
            }),
            capacity,
        )
    }
}

impl ChunkSizer for BertTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &impl ChunkCapacity) -> ChunkSize {
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
        let offsets = tokenizer.chunk_size(" An apple a", &capacity);
        assert_eq!(
            offsets,
            ChunkSize::from_offsets([1..3, 4..9, 10..11].into_iter(), &capacity)
        );
    }
}

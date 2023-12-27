use std::ops::Range;

use tiktoken_rs::CoreBPE;

use crate::{ChunkCapacity, ChunkSize, ChunkSizer};

impl ChunkSizer for CoreBPE {
    /// Returns the number of tokens in a given text after tokenization.
    fn chunk_size(&self, chunk: &str, capacity: &impl ChunkCapacity) -> ChunkSize {
        ChunkSize::from_offsets(encoded_offsets(self, chunk), capacity)
    }
}

impl ChunkSizer for &CoreBPE {
    /// Returns the number of tokens in a given text after tokenization.
    fn chunk_size(&self, chunk: &str, capacity: &impl ChunkCapacity) -> ChunkSize {
        ChunkSize::from_offsets(encoded_offsets(self, chunk), capacity)
    }
}

fn encoded_offsets<'text, 'bpe: 'text>(
    bpe: &'bpe CoreBPE,
    chunk: &'text str,
) -> impl Iterator<Item = Range<usize>> + 'text {
    let tokens = bpe.encode_ordinary(chunk);
    bpe._decode_native_and_split(tokens)
        .scan(0usize, |offset, bytes| {
            let end = *offset + bytes.len();
            let item = *offset..end;
            *offset = end;
            Some(item)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    use tiktoken_rs::cl100k_base;

    #[test]
    fn returns_offsets() {
        let offsets = encoded_offsets(&cl100k_base().unwrap(), "An apple a").collect::<Vec<_>>();
        assert_eq!(offsets, vec![0..2, 2..8, 8..10]);
    }
}

use tiktoken_rs::CoreBPE;

use crate::{ChunkSizer, EncodedOffsets};

impl ChunkSizer for CoreBPE {
    /// Return offsets for each unit of text used to calculate chunk size.
    /// Should return an exclusive byte range for each element counted.
    fn encoded_offsets(&self, chunk: &str) -> EncodedOffsets {
        encoded_offsets(self, chunk)
    }

    /// Returns the number of tokens in a given text after tokenization.
    ///
    /// # Panics
    ///
    /// Will panic if you don't have a byte-level tokenizer and the splitter
    /// encounters text it can't tokenize.
    fn chunk_size(&self, text: &str) -> usize {
        chunk_size(self, text)
    }
}

impl ChunkSizer for &CoreBPE {
    /// Return offsets for each unit of text used to calculate chunk size.
    /// Should return an exclusive byte range for each element counted.
    fn encoded_offsets(&self, chunk: &str) -> EncodedOffsets {
        encoded_offsets(self, chunk)
    }

    /// Returns the number of tokens in a given text after tokenization.
    ///
    /// # Panics
    ///
    /// Will panic if you don't have a byte-level tokenizer and the splitter
    /// encounters text it can't tokenize.
    fn chunk_size(&self, text: &str) -> usize {
        chunk_size(self, text)
    }
}

fn chunk_size(bpe: &CoreBPE, text: &str) -> usize {
    bpe.encode_ordinary(text).len()
}

fn encoded_offsets(bpe: &CoreBPE, chunk: &str) -> EncodedOffsets {
    let tokens = bpe.encode_ordinary(chunk);
    let decoded = bpe
        ._decode_native_and_split(tokens)
        .scan(0usize, |offset, bytes| {
            let end = *offset + bytes.len();
            let item = *offset..end;
            *offset = end;
            Some(item)
        });
    decoded.collect::<Vec<_>>().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    use tiktoken_rs::cl100k_base;

    #[test]
    fn returns_offsets() {
        let offsets = cl100k_base().unwrap().encoded_offsets("An apple a");
        assert_eq!(offsets, vec![0..2, 2..8, 8..10].into());
    }
}

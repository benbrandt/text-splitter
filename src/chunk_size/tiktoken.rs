use tiktoken_rs::CoreBPE;

use crate::ChunkSizer;

impl ChunkSizer for &CoreBPE {
    /// Returns the number of tokens in a given text after tokenization.
    fn size(&self, chunk: &str) -> usize {
        self.encode_ordinary(chunk).len()
    }
}

impl ChunkSizer for CoreBPE {
    /// Returns the number of tokens in a given text after tokenization.
    fn size(&self, chunk: &str) -> usize {
        (&self).size(chunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tiktoken_rs::cl100k_base;

    #[test]
    fn returns_offsets() {
        let tokenizer = cl100k_base().unwrap();
        let size = tokenizer.size("An apple a");
        assert_eq!(size, 3);
    }
}

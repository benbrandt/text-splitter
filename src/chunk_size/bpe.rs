use bpe_openai::Tokenizer;

use crate::ChunkSizer;

impl ChunkSizer for Tokenizer {
    /// Returns the number of tokens in a given text after tokenization.
    fn size(&self, chunk: &str) -> usize {
        self.count(chunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use bpe_openai::cl100k_base;

    #[test]
    fn returns_correct_token_count() {
        let tokenizer = cl100k_base();
        let size = tokenizer.size("An apple a");
        assert_eq!(size, 3);
    }
}
use crate::ChunkValidator;

/// Generic interface for tokenizers to calculate number of tokens.
pub trait TokenCount {
    /// Returns the number of tokens in a given text after tokenization.
    fn token_count(&self, text: &str) -> usize;
}

impl<T> ChunkValidator for T
where
    T: TokenCount,
{
    /// Determine if the given chunk still fits within the specified max chunk
    /// size, based on tokens.
    ///
    /// ```
    /// use text_splitter::{ChunkValidator};
    /// use tokenizers::Tokenizer;
    ///
    /// let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
    /// assert!(tokenizer.validate_chunk("hello", 10));
    /// ```
    fn validate_chunk(&self, chunk: &str, chunk_size: usize) -> bool {
        self.token_count(chunk) <= chunk_size
    }
}

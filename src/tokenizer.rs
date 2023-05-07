use crate::ChunkValidator;

/// Generic interface for tokenizers to calculate number of tokens.
pub trait TokenCount {
    /// Returns the number of tokens in a given text after tokenization.
    fn token_count(&self, text: &str) -> usize;
}

#[derive(Debug)]
/// Used for splitting a piece of text into chunks based on the number of
/// tokens in each chunk.
pub struct Tokens<T>
where
    T: TokenCount,
{
    /// Tokenizer to use in calculating number of tokens.
    tokenizer: T,
}

impl<T> Tokens<T>
where
    T: TokenCount,
{
    /// Creates a new [`Tokens`]. Chunks will be generated based on the
    /// number of tokens in the chunk.
    ///
    /// `tokenizer` is anything that implements [`TokenCount`].
    ///
    /// ```
    /// use text_splitter::{Characters, TextSplitter, Tokens};
    /// use tokenizers::Tokenizer;
    ///
    /// let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
    /// let splitter = TextSplitter::new(Tokens::new(tokenizer));
    /// ```
    pub fn new(tokenizer: T) -> Self {
        Self { tokenizer }
    }
}

impl<T> ChunkValidator for Tokens<T>
where
    T: TokenCount,
{
    /// Determine if the given chunk still fits within the specified max chunk
    /// size, based on tokens.
    ///
    /// ```
    /// use text_splitter::{ChunkValidator, Tokens};
    /// use tokenizers::Tokenizer;
    ///
    /// let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
    /// let tokens = Tokens::new(tokenizer);
    /// assert!(tokens.validate("hello", 10));
    /// ```
    fn validate(&self, chunk: &str, chunk_size: usize) -> bool {
        self.tokenizer.token_count(chunk) <= chunk_size
    }
}

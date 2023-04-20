use crate::ChunkSize;

/// Generic interface for tokenizers to calculate number of tokens.
pub trait NumTokens {
    /// Returns the number of tokens in a given text after tokenization.
    fn number_of_tokens(&self, text: &str) -> usize;
}

#[derive(Debug)]
/// Used for splitting a piece of text into chunks based on the number of
/// tokens in each chunk.
pub struct Tokens<T>
where
    T: NumTokens,
{
    /// Maximum size of a chunk, measured in tokens.
    max_tokens: usize,
    /// Tokenizer to use in calculating number of tokens.
    tokenizer: T,
}

impl<T> Tokens<T>
where
    T: NumTokens,
{
    /// Creates a new [`Tokens`]. Chunks will be generated based on the
    /// number of tokens in the chunk.
    ///
    /// `tokenizer` is anything that implements [`NumTokens`].
    /// `max_tokens` determines what the largest chunk will be.
    ///
    /// ```
    /// use text_splitter::{Characters, TextSplitter, Tokens};
    /// use tokenizers::Tokenizer;
    ///
    /// let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
    /// let splitter = TextSplitter::new(Tokens::new(tokenizer, 100));
    /// ```
    pub fn new(tokenizer: T, max_tokens: usize) -> Self {
        Self {
            max_tokens,
            tokenizer,
        }
    }
}

impl<T> ChunkSize for Tokens<T>
where
    T: NumTokens,
{
    /// Determine if the given chunk still fits within the specified max chunk
    /// size, based on tokens.
    ///
    /// ```
    /// use text_splitter::{ChunkSize, Tokens};
    /// use tokenizers::Tokenizer;
    ///
    /// let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
    /// let tokens = Tokens::new(tokenizer, 10);
    /// assert!(tokens.valid_chunk("hello"));
    /// ```
    fn valid_chunk(&self, chunk: &str) -> bool {
        self.tokenizer.number_of_tokens(chunk) <= self.max_tokens
    }
}

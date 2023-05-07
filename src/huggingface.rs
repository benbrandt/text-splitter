use tokenizers::Tokenizer;

use crate::TokenCount;

impl TokenCount for Tokenizer {
    /// Returns the number of tokens in a given text after tokenization.
    ///
    /// # Panics
    ///
    /// Will panic if you don't have a byte-level tokenizer and the splitter
    /// encounters text it can't tokenize.
    fn token_count(&self, text: &str) -> usize {
        self.encode(text, false)
            .map(|enc| enc.len())
            .expect("Unable to tokenize the following string {str}")
    }
}

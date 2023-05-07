use tiktoken_rs::CoreBPE;

use crate::TokenCount;

impl TokenCount for CoreBPE {
    /// Returns the number of tokens in a given text after tokenization.
    ///
    /// # Panics
    ///
    /// Will panic if you don't have a byte-level tokenizer and the splitter
    /// encounters text it can't tokenize.
    fn token_count(&self, text: &str) -> usize {
        self.encode_ordinary(text).len()
    }
}

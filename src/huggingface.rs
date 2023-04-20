use tokenizers::Tokenizer;

use crate::NumTokens;

impl NumTokens for Tokenizer {
    /// Returns the number of tokens in a given text after tokenization.
    ///
    /// # Panics
    ///
    /// Will panic if you don't have a byte-level tokenizer and the splitter
    /// encounters text it can't tokenize.
    fn number_of_tokens(&self, text: &str) -> usize {
        self.encode(text, false)
            .map(|enc| enc.len())
            .expect("Unable to tokenize the following string {str}")
    }
}

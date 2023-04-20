use tiktoken_rs::CoreBPE;

use crate::NumTokens;

impl NumTokens for CoreBPE {
    /// Returns the number of tokens in a given text after tokenization.
    ///
    /// # Panics
    ///
    /// Will panic if you don't have a byte-level tokenizer and the splitter
    /// encounters text it can't tokenize.
    fn number_of_tokens(&self, text: &str) -> usize {
        self.encode_ordinary(text).len()
    }
}

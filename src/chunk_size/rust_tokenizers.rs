use rust_tokenizers::tokenizer::{
    AlbertTokenizer, BertTokenizer, DeBERTaTokenizer, DeBERTaV2Tokenizer, FNetTokenizer,
    Gpt2Tokenizer, M2M100Tokenizer, MBart50Tokenizer, MarianTokenizer, NLLBTokenizer,
    OpenAiGptTokenizer, PegasusTokenizer, ProphetNetTokenizer, ReformerTokenizer, RobertaTokenizer,
    T5Tokenizer, Tokenizer, XLMRobertaTokenizer, XLNetTokenizer,
};
use rust_tokenizers::{Offset, TokensWithOffsets};
use std::ops::Range;

use crate::{ChunkCapacity, ChunkSize, ChunkSizer};

impl ChunkSizer for &AlbertTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        let tokens_with_offsets = self.tokenize_with_offsets(chunk);
        chunk_size_from_offsets(tokens_with_offsets, capacity)
    }
}

impl ChunkSizer for AlbertTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        (&self).chunk_size(chunk, capacity)
    }
}

impl ChunkSizer for &BertTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        let tokens_with_offsets = self.tokenize_with_offsets(chunk);
        chunk_size_from_offsets(tokens_with_offsets, capacity)
    }
}

impl ChunkSizer for BertTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        (&self).chunk_size(chunk, capacity)
    }
}
impl ChunkSizer for &DeBERTaTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        let tokens_with_offsets = self.tokenize_with_offsets(chunk);
        chunk_size_from_offsets(tokens_with_offsets, capacity)
    }
}

impl ChunkSizer for DeBERTaTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        (&self).chunk_size(chunk, capacity)
    }
}

impl ChunkSizer for &DeBERTaV2Tokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        let tokens_with_offsets = self.tokenize_with_offsets(chunk);
        chunk_size_from_offsets(tokens_with_offsets, capacity)
    }
}

impl ChunkSizer for DeBERTaV2Tokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        (&self).chunk_size(chunk, capacity)
    }
}

impl ChunkSizer for &FNetTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        let tokens_with_offsets = self.tokenize_with_offsets(chunk);
        chunk_size_from_offsets(tokens_with_offsets, capacity)
    }
}

impl ChunkSizer for FNetTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        (&self).chunk_size(chunk, capacity)
    }
}

impl ChunkSizer for &Gpt2Tokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        let tokens_with_offsets = self.tokenize_with_offsets(chunk);
        chunk_size_from_offsets(tokens_with_offsets, capacity)
    }
}

impl ChunkSizer for Gpt2Tokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        (&self).chunk_size(chunk, capacity)
    }
}

impl ChunkSizer for &M2M100Tokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        let tokens_with_offsets = self.tokenize_with_offsets(chunk);
        chunk_size_from_offsets(tokens_with_offsets, capacity)
    }
}

impl ChunkSizer for M2M100Tokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        (&self).chunk_size(chunk, capacity)
    }
}

impl ChunkSizer for &MBart50Tokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        let tokens_with_offsets = self.tokenize_with_offsets(chunk);
        chunk_size_from_offsets(tokens_with_offsets, capacity)
    }
}

impl ChunkSizer for MBart50Tokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        (&self).chunk_size(chunk, capacity)
    }
}

impl ChunkSizer for &MarianTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        let tokens_with_offsets = self.tokenize_with_offsets(chunk);
        chunk_size_from_offsets(tokens_with_offsets, capacity)
    }
}

impl ChunkSizer for MarianTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        (&self).chunk_size(chunk, capacity)
    }
}

impl ChunkSizer for &NLLBTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        let tokens_with_offsets = self.tokenize_with_offsets(chunk);
        chunk_size_from_offsets(tokens_with_offsets, capacity)
    }
}

impl ChunkSizer for NLLBTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        (&self).chunk_size(chunk, capacity)
    }
}

impl ChunkSizer for &OpenAiGptTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        let tokens_with_offsets = self.tokenize_with_offsets(chunk);
        chunk_size_from_offsets(tokens_with_offsets, capacity)
    }
}

impl ChunkSizer for OpenAiGptTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        (&self).chunk_size(chunk, capacity)
    }
}
impl ChunkSizer for &PegasusTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        let tokens_with_offsets = self.tokenize_with_offsets(chunk);
        chunk_size_from_offsets(tokens_with_offsets, capacity)
    }
}

impl ChunkSizer for PegasusTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        (&self).chunk_size(chunk, capacity)
    }
}
impl ChunkSizer for &ProphetNetTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        let tokens_with_offsets = self.tokenize_with_offsets(chunk);
        chunk_size_from_offsets(tokens_with_offsets, capacity)
    }
}

impl ChunkSizer for ProphetNetTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        (&self).chunk_size(chunk, capacity)
    }
}
impl ChunkSizer for &ReformerTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        let tokens_with_offsets = self.tokenize_with_offsets(chunk);
        chunk_size_from_offsets(tokens_with_offsets, capacity)
    }
}

impl ChunkSizer for ReformerTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        (&self).chunk_size(chunk, capacity)
    }
}
impl ChunkSizer for &XLNetTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        let tokens_with_offsets = self.tokenize_with_offsets(chunk);
        chunk_size_from_offsets(tokens_with_offsets, capacity)
    }
}

impl ChunkSizer for XLNetTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        (&self).chunk_size(chunk, capacity)
    }
}
impl ChunkSizer for &RobertaTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        let tokens_with_offsets = self.tokenize_with_offsets(chunk);
        chunk_size_from_offsets(tokens_with_offsets, capacity)
    }
}

impl ChunkSizer for RobertaTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        (&self).chunk_size(chunk, capacity)
    }
}

impl ChunkSizer for &T5Tokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        let tokens_with_offsets = self.tokenize_with_offsets(chunk);
        chunk_size_from_offsets(tokens_with_offsets, capacity)
    }
}

impl ChunkSizer for T5Tokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        (&self).chunk_size(chunk, capacity)
    }
}

impl ChunkSizer for &XLMRobertaTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        let tokens_with_offsets = self.tokenize_with_offsets(chunk);
        chunk_size_from_offsets(tokens_with_offsets, capacity)
    }
}

impl ChunkSizer for XLMRobertaTokenizer {
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        (&self).chunk_size(chunk, capacity)
    }
}

fn chunk_size_from_offsets(
    tokens_with_offsets: TokensWithOffsets,
    capacity: &ChunkCapacity,
) -> ChunkSize {
    let offsets = tokens_with_offsets
        .offsets
        .iter()
        .flatten()
        .map(|Offset { begin, end }| Range {
            start: *begin as usize,
            end: *end as usize,
        });
    ChunkSize::from_offsets(offsets, capacity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_offsets() {
        let path: &str = "tests/tokenizers/bert-uncased-vocab.txt";
        let tokenizer: BertTokenizer = BertTokenizer::from_file(path, false, false).unwrap();
        let capacity = 10;
        let offsets = tokenizer.chunk_size(" An apple a", &capacity.into());
        assert_eq!(
            offsets,
            ChunkSize::from_offsets([1..3, 4..9, 10..11].into_iter(), &capacity.into())
        );
    }
}

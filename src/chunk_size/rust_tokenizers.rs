use std::ops::Range;

use rust_tokenizers::{
    tokenizer::{
        AlbertTokenizer, BaseTokenizer, BertTokenizer, CtrlTokenizer, DeBERTaTokenizer,
        DeBERTaV2Tokenizer, FNetTokenizer, Gpt2Tokenizer, M2M100Tokenizer, MBart50Tokenizer,
        MarianTokenizer, NLLBTokenizer, OpenAiGptTokenizer, PegasusTokenizer, ProphetNetTokenizer,
        ReformerTokenizer, RobertaTokenizer, T5Tokenizer, Tokenizer, XLMRobertaTokenizer,
        XLNetTokenizer,
    },
    vocab::Vocab,
    Offset,
};

use crate::{ChunkCapacity, ChunkSize, ChunkSizer};

fn chunk_size_from_offsets<V: Vocab, T: Tokenizer<V>>(
    tokenizer: &T,
    chunk: &str,
    capacity: &ChunkCapacity,
) -> ChunkSize {
    let tokens_with_offsets = tokenizer.tokenize_with_offsets(chunk);
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

impl<V> ChunkSizer for &BaseTokenizer<V>
where
    V: Vocab + Sync + Send,
{
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        chunk_size_from_offsets(*self, chunk, capacity)
    }
}

impl<V> ChunkSizer for BaseTokenizer<V>
where
    V: Vocab + Sync + Send,
{
    fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
        (&self).chunk_size(chunk, capacity)
    }
}

macro_rules! impl_chunk_sizer {
    ($($t:ty),+) => {
        $(impl ChunkSizer for &$t {
            fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
                chunk_size_from_offsets(*self, chunk, capacity)
            }
        }

        impl ChunkSizer for $t {
            fn chunk_size(&self, chunk: &str, capacity: &ChunkCapacity) -> ChunkSize {
                (&self).chunk_size(chunk, capacity)
            }
        })+
    }
}

impl_chunk_sizer!(
    AlbertTokenizer,
    BertTokenizer,
    CtrlTokenizer,
    DeBERTaTokenizer,
    DeBERTaV2Tokenizer,
    FNetTokenizer,
    Gpt2Tokenizer,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MarianTokenizer,
    NLLBTokenizer,
    OpenAiGptTokenizer,
    PegasusTokenizer,
    ProphetNetTokenizer,
    ReformerTokenizer,
    RobertaTokenizer,
    T5Tokenizer,
    XLMRobertaTokenizer,
    XLNetTokenizer
);

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use cached_path::Cache;

    use super::*;

    /// Downloads a remote file to the cache directory if it doensn't already exist,
    /// and returns the path to the cached file.
    fn download_file_to_cache(src: &str) -> PathBuf {
        let mut cache_dir = dirs::home_dir().unwrap();
        cache_dir.push(".cache");
        cache_dir.push(".text-splitter");

        Cache::builder()
            .dir(cache_dir)
            .build()
            .unwrap()
            .cached_path(src)
            .unwrap()
    }

    #[test]
    fn returns_offsets() {
        let vocab_path = download_file_to_cache(
            "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
        );
        let tokenizer: BertTokenizer = BertTokenizer::from_file(vocab_path, false, false).unwrap();
        let capacity = 10;
        let offsets = tokenizer.chunk_size(" An apple a", &capacity.into());
        assert_eq!(
            offsets,
            ChunkSize::from_offsets([1..3, 4..9, 10..11].into_iter(), &capacity.into())
        );
    }
}

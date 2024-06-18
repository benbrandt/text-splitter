use rust_tokenizers::{
    tokenizer::{
        AlbertTokenizer, BaseTokenizer, BertTokenizer, CtrlTokenizer, DeBERTaTokenizer,
        DeBERTaV2Tokenizer, FNetTokenizer, Gpt2Tokenizer, M2M100Tokenizer, MBart50Tokenizer,
        MarianTokenizer, NLLBTokenizer, OpenAiGptTokenizer, PegasusTokenizer, ProphetNetTokenizer,
        ReformerTokenizer, RobertaTokenizer, SentencePieceBpeTokenizer, SentencePieceTokenizer,
        T5Tokenizer, Tokenizer, XLMRobertaTokenizer, XLNetTokenizer,
    },
    vocab::Vocab,
};

use crate::ChunkSizer;

fn chunk_size_from_offsets<V: Vocab, T: Tokenizer<V>>(tokenizer: &T, chunk: &str) -> usize {
    tokenizer.tokenize(chunk).len()
}

impl<V> ChunkSizer for &BaseTokenizer<V>
where
    V: Vocab + Sync + Send,
{
    fn size(&self, chunk: &str) -> usize {
        chunk_size_from_offsets(*self, chunk)
    }
}

impl<V> ChunkSizer for BaseTokenizer<V>
where
    V: Vocab + Sync + Send,
{
    fn size(&self, chunk: &str) -> usize {
        (&self).size(chunk)
    }
}

macro_rules! impl_chunk_sizer {
    ($($t:ty),+) => {
        $(impl ChunkSizer for &$t {
            fn size(&self, chunk: &str) -> usize {
                chunk_size_from_offsets(*self, chunk)
            }
        }

        impl ChunkSizer for $t {
            fn size(&self, chunk: &str) -> usize {
                (&self).size(chunk)
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
    SentencePieceBpeTokenizer,
    SentencePieceTokenizer,
    T5Tokenizer,
    XLMRobertaTokenizer,
    XLNetTokenizer
);

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use cached_path::Cache;
    use rayon::prelude::*;
    use rust_tokenizers::vocab::{BertVocab, BpePairVocab, Gpt2Vocab, ProphetNetVocab};
    use strum::{EnumIter, IntoEnumIterator};

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
        let tokenizer = BertTokenizer::from_file(vocab_path, false, false).unwrap();
        let size = tokenizer.size(" An apple a");
        assert_eq!(size, 3);
    }

    #[test]
    fn smoke_test() {
        let sizes = TokenizerOption::iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|tokenizer| tokenizer.tokenizer().size(" An apple a"));
        assert!(sizes.all(|size| size > 0));
    }

    #[derive(EnumIter)]
    enum TokenizerOption {
        Albert,
        Base,
        Bert,
        Ctrl,
        DeBERTa,
        DeBERTaV2,
        FNet,
        Gpt2,
        M2M100,
        MBart50,
        // Marian, // No example source vocab at the moment
        Nllb,
        OpenAiGpt,
        Pegasus,
        ProphetNet,
        Reformer,
        Roberta,
        SentencePieceBpe,
        SentencePiece,
        T5,
        XLMRoberta,
        XLNet,
    }

    impl TokenizerOption {
        #[allow(clippy::too_many_lines)]
        fn tokenizer(&self) -> Box<dyn ChunkSizer> {
            match self {
                Self::Albert => {
                    let vocab_path = download_file_to_cache(
                        "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v2-spiece.model",
                    );
                    Box::new(AlbertTokenizer::from_file(vocab_path, false, false).unwrap())
                }
                Self::Base => {
                    let vocab_path = download_file_to_cache(
                        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
                    );
                    let vocab = BertVocab::from_file(vocab_path).unwrap();
                    Box::new(BaseTokenizer::from_existing_vocab(vocab, false, false))
                }
                Self::Bert => {
                    let vocab_path = download_file_to_cache(
                        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
                    );
                    Box::new(BertTokenizer::from_file(vocab_path, false, false).unwrap())
                }
                Self::Ctrl => {
                    let vocab_path = download_file_to_cache(
                        "https://raw.githubusercontent.com/salesforce/ctrl/master/ctrl-vocab.json",
                    );
                    let merges_path = download_file_to_cache(
                        "https://raw.githubusercontent.com/salesforce/ctrl/master/ctrl-merges.txt",
                    );
                    Box::new(CtrlTokenizer::from_file(vocab_path, merges_path, false).unwrap())
                }
                Self::DeBERTa => {
                    let vocab_path = download_file_to_cache(
                        "https://huggingface.co/microsoft/deberta-base/resolve/main/vocab.json",
                    );
                    let merges_path = download_file_to_cache(
                        "https://huggingface.co/microsoft/deberta-base/resolve/main/merges.txt",
                    );
                    Box::new(DeBERTaTokenizer::from_file(vocab_path, merges_path, false).unwrap())
                }
                Self::DeBERTaV2 => {
                    let vocab_path = download_file_to_cache(
                        "https://huggingface.co/microsoft/deberta-v3-base/resolve/main/spm.model",
                    );
                    Box::new(
                        DeBERTaV2Tokenizer::from_file(vocab_path, false, false, false).unwrap(),
                    )
                }
                Self::FNet => {
                    let vocab_path = download_file_to_cache(
                        "https://huggingface.co/google/fnet-base/resolve/main/spiece.model",
                    );
                    Box::new(FNetTokenizer::from_file(vocab_path, false, false).unwrap())
                }
                Self::Gpt2 => {
                    let vocab_path = download_file_to_cache(
                        "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",
                    );
                    let merges_path = download_file_to_cache(
                        "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",
                    );
                    let vocab = Gpt2Vocab::from_file(vocab_path.as_path()).unwrap();
                    let merges = BpePairVocab::from_file(merges_path.as_path()).unwrap();

                    Box::new(Gpt2Tokenizer::from_existing_vocab_and_merges(
                        vocab, merges, false,
                    ))
                }
                Self::M2M100 => {
                    let vocab_path = download_file_to_cache(
                        "https://huggingface.co/facebook/m2m100_418M/resolve/main/vocab.json",
                    );
                    let merges_path = download_file_to_cache(
                        "https://huggingface.co/facebook/m2m100_418M/resolve/main/sentencepiece.bpe.model",
                    );

                    Box::new(M2M100Tokenizer::from_files(vocab_path, merges_path, false).unwrap())
                }
                Self::MBart50 => {
                    let vocab_path = download_file_to_cache(
                        "https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/resolve/main/sentencepiece.bpe.model",
                    );

                    Box::new(MBart50Tokenizer::from_file(vocab_path, false).unwrap())
                }
                Self::Nllb => {
                    let vocab_path = download_file_to_cache(
                        "https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/main/tokenizer.json",
                    );
                    let merges_path = download_file_to_cache(
                        "https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/main/sentencepiece.bpe.model",
                    );
                    let special_path = download_file_to_cache(
                        "https://huggingface.co/facebook/nllb-200-distilled-600M/raw/main/special_tokens_map.json",
                    );

                    Box::new(
                        NLLBTokenizer::from_files_with_special_token_map(
                            vocab_path,
                            merges_path,
                            special_path,
                        )
                        .unwrap(),
                    )
                }
                Self::OpenAiGpt => {
                    let vocab_path = download_file_to_cache(
                        "https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-vocab.json",
                    );
                    let merges_path = download_file_to_cache(
                        "https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-merges.txt",
                    );

                    Box::new(OpenAiGptTokenizer::from_file(vocab_path, merges_path, true).unwrap())
                }
                Self::Pegasus => {
                    let vocab_path = download_file_to_cache(
                        "https://cdn.huggingface.co/google/pegasus-cnn_dailymail/spiece.model",
                    );

                    Box::new(PegasusTokenizer::from_file(vocab_path, false).unwrap())
                }
                Self::ProphetNet => {
                    let vocab_path = download_file_to_cache(
                        "https://huggingface.co/microsoft/prophetnet-large-uncased/resolve/main/prophetnet.tokenizer",
                    );
                    let vocab = ProphetNetVocab::from_file(vocab_path).unwrap();

                    Box::new(ProphetNetTokenizer::from_existing_vocab(vocab, true, true))
                }
                Self::Reformer => {
                    let vocab_path = download_file_to_cache(
                        "https://cdn.huggingface.co/google/reformer-crime-and-punishment/spiece.model",
                    );

                    Box::new(ReformerTokenizer::from_file(vocab_path, false).unwrap())
                }
                Self::Roberta => {
                    let vocab_path = download_file_to_cache(
                        "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json",
                    );
                    let merges_path = download_file_to_cache(
                        "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt",
                    );

                    Box::new(
                        RobertaTokenizer::from_file(vocab_path, merges_path, false, true).unwrap(),
                    )
                }
                Self::SentencePieceBpe => {
                    let vocab_path = download_file_to_cache(
                        "https://huggingface.co/facebook/m2m100_418M/resolve/main/sentencepiece.bpe.model",
                    );

                    Box::new(SentencePieceBpeTokenizer::from_file(vocab_path, false).unwrap())
                }

                Self::SentencePiece => {
                    let vocab_path = download_file_to_cache(
                        "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-spiece.model",
                    );

                    Box::new(SentencePieceTokenizer::from_file(vocab_path, false).unwrap())
                }
                Self::T5 => {
                    let vocab_path = download_file_to_cache(
                        "https://huggingface.co/t5-base/resolve/main/spiece.model",
                    );

                    Box::new(T5Tokenizer::from_file(vocab_path, false).unwrap())
                }
                Self::XLMRoberta => {
                    let vocab_path = download_file_to_cache("https://cdn.huggingface.co/xlm-roberta-large-finetuned-conll03-english-sentencepiece.bpe.model");

                    Box::new(XLMRobertaTokenizer::from_file(vocab_path, false).unwrap())
                }
                Self::XLNet => {
                    let vocab_path = download_file_to_cache(
                        "https://cdn.huggingface.co/xlnet-base-cased-spiece.model",
                    );

                    Box::new(XLNetTokenizer::from_file(vocab_path, false, true).unwrap())
                }
            }
        }
    }
}

use std::fs;

use once_cell::sync::Lazy;
use text_splitter::{TextSplitter, Tokens};
use tiktoken_rs::{cl100k_base, CoreBPE};

static TOKENIZER: Lazy<CoreBPE> = Lazy::new(|| cl100k_base().unwrap());

#[test]
fn default() {
    insta::glob!("texts/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in [10, 100, 1000] {
            let splitter = TextSplitter::new(Tokens::new(TOKENIZER.clone(), chunk_size));
            let chunks = splitter.chunk_by_paragraphs(&text).collect::<Vec<_>>();

            assert_eq!(chunks.join(""), text);
            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

#[test]
fn trim() {
    insta::glob!("texts/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in [10, 100, 1000] {
            let splitter = TextSplitter::new(Tokens::new(TOKENIZER.clone(), chunk_size))
                .with_trim_chunks(true);
            let chunks = splitter.chunk_by_paragraphs(&text).collect::<Vec<_>>();

            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

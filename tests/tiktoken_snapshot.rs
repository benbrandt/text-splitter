use std::fs;

use once_cell::sync::Lazy;
use text_splitter::TextSplitter;
use tiktoken_rs::{cl100k_base, CoreBPE};

static TOKENIZER: Lazy<CoreBPE> = Lazy::new(|| cl100k_base().unwrap());

#[test]
fn default() {
    insta::glob!("texts/*.txt", |path| {
        let text = fs::read_to_string(path).unwrap();

        for chunk_size in [10, 100, 1000] {
            let splitter = TextSplitter::new(TOKENIZER.clone());
            let chunks = splitter.chunks(&text, chunk_size).collect::<Vec<_>>();

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
            let splitter = TextSplitter::new(TOKENIZER.clone()).with_trim_chunks(true);
            let chunks = splitter.chunks(&text, chunk_size).collect::<Vec<_>>();

            insta::assert_yaml_snapshot!(chunks);
        }
    });
}

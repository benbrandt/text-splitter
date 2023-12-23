use std::fs;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use text_splitter::{Characters, TextSplitter};
use tiktoken_rs::{cl100k_base, CoreBPE};
use tokenizers::Tokenizer;

#[allow(clippy::large_enum_variant)]
enum Splitter {
    Characters(TextSplitter<Characters>),
    Huggingface(TextSplitter<Tokenizer>),
    Tiktoken(TextSplitter<CoreBPE>),
}

impl Splitter {
    fn name(&self) -> &str {
        match self {
            Splitter::Characters(_) => "Characters",
            Splitter::Huggingface(_) => "Huggingface",
            Splitter::Tiktoken(_) => "Tiktoken",
        }
    }

    fn iter() -> [Self; 3] {
        [
            Self::Characters(TextSplitter::default()),
            Self::Huggingface(TextSplitter::new(
                Tokenizer::from_pretrained("bert-base-cased", None).unwrap(),
            )),
            Self::Tiktoken(TextSplitter::new(cl100k_base().unwrap())),
        ]
    }

    fn chunks<'text>(&self, text: &'text str, chunk_size: usize) -> Vec<&'text str> {
        match self {
            Splitter::Characters(splitter) => splitter.chunks(text, chunk_size).collect(),
            Splitter::Huggingface(splitter) => splitter.chunks(text, chunk_size).collect(),
            Splitter::Tiktoken(splitter) => splitter.chunks(text, chunk_size).collect(),
        }
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    for filename in ["romeo_and_juliet", "room_with_a_view"] {
        let mut group = c.benchmark_group(filename);
        let text = fs::read_to_string(format!("tests/inputs/text/{filename}.txt")).unwrap();

        for splitter in Splitter::iter() {
            for chunk_size in (5..17).map(|n| 2usize.pow(n)) {
                group.bench_with_input(
                    BenchmarkId::new(splitter.name(), chunk_size),
                    &chunk_size,
                    |b, &chunk_size| b.iter(|| splitter.chunks(&text, chunk_size)),
                );
            }
        }
        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

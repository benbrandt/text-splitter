#![allow(missing_docs)]

use std::fs;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use text_splitter::{Characters, MarkdownSplitter, TextSplitter};
use tiktoken_rs::{cl100k_base, CoreBPE};
use tokenizers::Tokenizer;

#[allow(clippy::large_enum_variant)]
enum TextSplitterImpl {
    Characters(TextSplitter<Characters>),
    Huggingface(TextSplitter<Tokenizer>),
    Tiktoken(TextSplitter<CoreBPE>),
}

impl TextSplitterImpl {
    fn name(&self) -> &str {
        match self {
            TextSplitterImpl::Characters(_) => "Characters",
            TextSplitterImpl::Huggingface(_) => "Huggingface",
            TextSplitterImpl::Tiktoken(_) => "Tiktoken",
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
            Self::Characters(splitter) => splitter.chunks(text, chunk_size).collect(),
            Self::Huggingface(splitter) => splitter.chunks(text, chunk_size).collect(),
            Self::Tiktoken(splitter) => splitter.chunks(text, chunk_size).collect(),
        }
    }
}

#[allow(clippy::large_enum_variant)]
enum MarkdownSplitterImpl {
    Characters(MarkdownSplitter<Characters>),
    Huggingface(MarkdownSplitter<Tokenizer>),
    Tiktoken(MarkdownSplitter<CoreBPE>),
}

impl MarkdownSplitterImpl {
    fn name(&self) -> &str {
        match self {
            MarkdownSplitterImpl::Characters(_) => "Characters",
            MarkdownSplitterImpl::Huggingface(_) => "Huggingface",
            MarkdownSplitterImpl::Tiktoken(_) => "Tiktoken",
        }
    }

    fn iter() -> [Self; 3] {
        [
            Self::Characters(MarkdownSplitter::default()),
            Self::Huggingface(MarkdownSplitter::new(
                Tokenizer::from_pretrained("bert-base-cased", None).unwrap(),
            )),
            Self::Tiktoken(MarkdownSplitter::new(cl100k_base().unwrap())),
        ]
    }

    fn chunks<'text>(&self, text: &'text str, chunk_size: usize) -> Vec<&'text str> {
        match self {
            Self::Characters(splitter) => splitter.chunks(text, chunk_size).collect(),
            Self::Huggingface(splitter) => splitter.chunks(text, chunk_size).collect(),
            Self::Tiktoken(splitter) => splitter.chunks(text, chunk_size).collect(),
        }
    }
}

fn text_benchmark(c: &mut Criterion) {
    for filename in ["romeo_and_juliet", "room_with_a_view"] {
        let mut group = c.benchmark_group(filename);
        let text = fs::read_to_string(format!("tests/inputs/text/{filename}.txt")).unwrap();

        for splitter in TextSplitterImpl::iter() {
            for chunk_size in (2..9).map(|n| 4usize.pow(n)) {
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

fn markdown_benchmark(c: &mut Criterion) {
    for filename in ["commonmark_spec"] {
        let mut group = c.benchmark_group(filename);
        let text = fs::read_to_string(format!("tests/inputs/markdown/{filename}.md")).unwrap();

        for splitter in MarkdownSplitterImpl::iter() {
            for chunk_size in (2..9).map(|n| 4usize.pow(n)) {
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

criterion_group!(benches, text_benchmark, markdown_benchmark);
criterion_main!(benches);

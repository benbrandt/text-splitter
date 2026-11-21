use crate::{ChunkSize, ChunkSizer};

/// Used for splitting a piece of text into chunks based on the number of
/// characters in each chunk.
///
/// ```
/// use text_splitter::TextSplitter;
///
/// // Uses character splitter by default.
/// let splitter = TextSplitter::new(10);
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Characters;

impl ChunkSizer for Characters {
    /// Determine the size of a given chunk to use for validation.
    fn size(&self, chunk: &str) -> usize {
        chunk.chars().count()
    }

    fn size_with_limit(&self, chunk: &str, limit: usize) -> ChunkSize {
        // Each Unicode scalar occupies at most four UTF-8 bytes.
        if chunk.len().div_ceil(4) > limit {
            ChunkSize::ExceedsLimit
        } else {
            ChunkSize::Exact(self.size(chunk))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_size() {
        let offsets = Characters.size("eé");
        assert_eq!(offsets, 2);
    }

    #[test]
    fn budget_is_inclusive_and_counts_unicode_scalars_not_bytes() {
        for text in ["", "x", "é", "東京", "🦀🦀", "e\u{301}🦀"] {
            let size = Characters.size(text);
            for limit in [0, 1, size.saturating_sub(1), size, size + 1, usize::MAX] {
                match Characters.size_with_limit(text, limit) {
                    ChunkSize::Exact(n) => assert_eq!(n, size),
                    ChunkSize::ExceedsLimit => assert!(size > limit),
                }
            }
        }
        assert_eq!(Characters.size_with_limit("🦀", 1), ChunkSize::Exact(1));
        assert_eq!(
            Characters.size_with_limit("🦀x", 1),
            ChunkSize::ExceedsLimit
        );
    }
}

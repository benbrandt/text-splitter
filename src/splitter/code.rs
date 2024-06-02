use std::{cmp::Ordering, ops::Range};

use thiserror::Error;
use tree_sitter::{Language, LanguageError, Parser, TreeCursor, MIN_COMPATIBLE_LANGUAGE_VERSION};

use crate::{
    splitter::{SemanticLevel, Splitter},
    trim::Trim,
    ChunkConfig, ChunkSizer,
};

/// Indicates there was an error with creating a `CodeSplitter`.
/// The `Display` implementation will provide a human-readable error message to
/// help debug the issue that caused the error.
#[derive(Error, Debug)]
#[error(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct CodeSplitterError(#[from] CodeSplitterErrorRepr);

/// Private error and free to change across minor version of the crate.
#[derive(Error, Debug)]
enum CodeSplitterErrorRepr {
    #[error(
        "Language version {0:?} is too old. Expected at least version {}",
        MIN_COMPATIBLE_LANGUAGE_VERSION
    )]
    LanguageError(LanguageError),
}

/// Source code splitter. Recursively splits chunks into the largest
/// semantic units that fit within the chunk size. Also will attempt to merge
/// neighboring chunks if they can fit within the given chunk size.
#[derive(Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct CodeSplitter<Sizer>
where
    Sizer: ChunkSizer,
{
    /// Method of determining chunk sizes.
    chunk_config: ChunkConfig<Sizer>,
    /// Language to use for parsing the code.
    language: Language,
}

impl<Sizer> CodeSplitter<Sizer>
where
    Sizer: ChunkSizer,
{
    /// Creates a new [`CodeSplitter`].
    ///
    /// ```
    /// use text_splitter::CodeSplitter;
    ///
    /// // By default, the chunk sizer is based on characters.
    /// let splitter = CodeSplitter::new(tree_sitter_rust::language(), 512).expect("Invalid language");
    /// ```
    ///
    /// # Errors
    ///
    /// Will return an error if the language version is too old to be compatible
    /// with the current version of the tree-sitter crate.
    pub fn new(
        language: Language,
        chunk_config: impl Into<ChunkConfig<Sizer>>,
    ) -> Result<Self, CodeSplitterError> {
        // Verify that this is a valid language so we can rely on that later.
        let mut parser = Parser::new();
        parser
            .set_language(&language)
            .map_err(CodeSplitterErrorRepr::LanguageError)?;
        Ok(Self {
            chunk_config: chunk_config.into(),
            language,
        })
    }

    /// Generate a list of chunks from a given text. Each chunk will be up to the `chunk_capacity`.
    ///
    /// ## Method
    ///
    /// To preserve as much semantic meaning within a chunk as possible, each chunk is composed of the largest semantic units that can fit in the next given chunk. For each splitter type, there is a defined set of semantic levels. Here is an example of the steps used:
    //
    // 1. Split the text by a increasing semantic levels.
    // 2. Check the first item for each level and select the highest level whose first item still fits within the chunk size.
    // 3. Merge as many of these neighboring sections of this level or above into a chunk to maximize chunk length.
    //    Boundaries of higher semantic levels are always included when merging, so that the chunk doesn't inadvertantly cross semantic boundaries.
    //
    // The boundaries used to split the text if using the `chunks` method, in ascending order:
    //
    // 1. Characters
    // 2. [Unicode Grapheme Cluster Boundaries](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
    // 3. [Unicode Word Boundaries](https://www.unicode.org/reports/tr29/#Word_Boundaries)
    // 4. [Unicode Sentence Boundaries](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
    // 5. Ascending depth of the syntax tree. So function would have a higher level than a statement inside of the function, and so on.
    //
    // Splitting doesn't occur below the character level, otherwise you could get partial bytes of a char, which may not be a valid unicode str.
    ///
    /// ```
    /// use text_splitter::CodeSplitter;
    ///
    /// let splitter = CodeSplitter::new(tree_sitter_rust::language(), 10).expect("Invalid language");
    /// let text = "Some text\n\nfrom a\ndocument";
    /// let chunks = splitter.chunks(text).collect::<Vec<_>>();
    ///
    /// assert_eq!(vec!["Some text", "from a", "document"], chunks);
    /// ```
    pub fn chunks<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
    ) -> impl Iterator<Item = &'text str> + 'splitter {
        Splitter::<_>::chunks(self, text)
    }

    /// Returns an iterator over chunks of the text and their byte offsets.
    /// Each chunk will be up to the `chunk_capacity`.
    ///
    /// See [`CodeSplitter::chunks`] for more information.
    ///
    /// ```
    /// use text_splitter::CodeSplitter;
    ///
    /// let splitter = CodeSplitter::new(tree_sitter_rust::language(), 10).expect("Invalid language");
    /// let text = "Some text\n\nfrom a\ndocument";
    /// let chunks = splitter.chunk_indices(text).collect::<Vec<_>>();
    ///
    /// assert_eq!(vec![(0, "Some text"), (11, "from a"), (18, "document")], chunks);
    pub fn chunk_indices<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
    ) -> impl Iterator<Item = (usize, &'text str)> + 'splitter {
        Splitter::<_>::chunk_indices(self, text)
    }
}

impl<Sizer> Splitter<Sizer> for CodeSplitter<Sizer>
where
    Sizer: ChunkSizer,
{
    type Level = Depth;

    const TRIM: Trim = Trim::PreserveIndentation;

    fn chunk_config(&self) -> &ChunkConfig<Sizer> {
        &self.chunk_config
    }

    fn parse(&self, text: &str) -> Vec<(Self::Level, Range<usize>)> {
        let mut parser = Parser::new();
        parser
            .set_language(&self.language)
            // We verify at initialization that the language is valid, so this should be safe.
            .expect("Error loading language");
        // The only reason the tree would be None is:
        // - No language was set (we do that)
        // - There was a timeout or cancellation option set (we don't)
        // - So it should be safe to unwrap here
        let tree = parser.parse(text, None).expect("Error parsing source code");

        CursorOffsets::new(tree.walk()).collect()
    }
}

/// New type around a usize to capture the depth of a given code node.
/// Custom type so that we can implement custom ordering, since we want to
/// sort items of lower depth as higher priority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Depth(usize);

impl PartialOrd for Depth {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Depth {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.cmp(&self.0)
    }
}

/// New type around a tree-sitter cursor to allow for implementing an iterator.
/// Each call to `next()` will return the next node in the tree in a depth-first
/// order.
struct CursorOffsets<'cursor> {
    cursor: TreeCursor<'cursor>,
}

impl<'cursor> CursorOffsets<'cursor> {
    fn new(cursor: TreeCursor<'cursor>) -> Self {
        Self { cursor }
    }
}

impl<'cursor> Iterator for CursorOffsets<'cursor> {
    type Item = (Depth, Range<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        // There are children (can call this initially because we don't want the root node)
        if self.cursor.goto_first_child() {
            return Some((
                Depth(self.cursor.depth() as usize),
                self.cursor.node().byte_range(),
            ));
        }

        loop {
            // There are sibling elements to grab
            if self.cursor.goto_next_sibling() {
                return Some((
                    Depth(self.cursor.depth() as usize),
                    self.cursor.node().byte_range(),
                ));
            // Start going back up the tree and check for next sibling on next iteration.
            } else if self.cursor.goto_parent() {
                continue;
            }

            // We have no more siblings or parents, so we're done.
            return None;
        }
    }
}

impl SemanticLevel for Depth {}

#[cfg(test)]
mod tests {
    use tree_sitter::{Node, Tree};

    use super::*;

    #[test]
    fn rust_splitter() {
        let splitter = CodeSplitter::new(tree_sitter_rust::language(), 16).unwrap();
        let text = "fn main() {\n    let x = 5;\n}";
        let chunks = splitter.chunks(text).collect::<Vec<_>>();

        assert_eq!(chunks, vec!["fn main()", "{\n    let x = 5;", "}"]);
    }

    #[test]
    fn rust_splitter_indices() {
        let splitter = CodeSplitter::new(tree_sitter_rust::language(), 16).unwrap();
        let text = "fn main() {\n    let x = 5;\n}";
        let chunks = splitter.chunk_indices(text).collect::<Vec<_>>();

        assert_eq!(
            chunks,
            vec![(0, "fn main()"), (10, "{\n    let x = 5;"), (27, "}")]
        );
    }

    #[test]
    fn depth_partialord() {
        assert_eq!(Depth(0).partial_cmp(&Depth(1)), Some(Ordering::Greater));
        assert_eq!(Depth(1).partial_cmp(&Depth(2)), Some(Ordering::Greater));
        assert_eq!(Depth(1).partial_cmp(&Depth(1)), Some(Ordering::Equal));
        assert_eq!(Depth(2).partial_cmp(&Depth(1)), Some(Ordering::Less));
    }

    #[test]
    fn depth_ord() {
        assert_eq!(Depth(0).cmp(&Depth(1)), Ordering::Greater);
        assert_eq!(Depth(1).cmp(&Depth(2)), Ordering::Greater);
        assert_eq!(Depth(1).cmp(&Depth(1)), Ordering::Equal);
        assert_eq!(Depth(2).cmp(&Depth(1)), Ordering::Less);
    }

    #[test]
    fn depth_sorting() {
        let mut depths = vec![Depth(0), Depth(1), Depth(2)];
        depths.sort();
        assert_eq!(depths, [Depth(2), Depth(1), Depth(0)]);
    }

    /// Checks that the optimized version of the code produces the same results as the naive version.
    #[test]
    fn optimized_code_offsets() {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_rust::language())
            .expect("Error loading Rust grammar");
        let source_code = "fn test() {
    let x = 1;
}";
        let tree = parser
            .parse(source_code, None)
            .expect("Error parsing source code");

        let offsets = CursorOffsets::new(tree.walk()).collect::<Vec<_>>();

        assert_eq!(offsets, naive_offsets(&tree));
    }

    #[test]
    fn multiple_top_siblings() {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_rust::language())
            .expect("Error loading Rust grammar");
        let source_code = "
fn fn1() {}
fn fn2() {}
fn fn3() {}
fn fn4() {}";
        let tree = parser
            .parse(source_code, None)
            .expect("Error parsing source code");

        let offsets = CursorOffsets::new(tree.walk()).collect::<Vec<_>>();

        assert_eq!(offsets, naive_offsets(&tree));
    }

    fn naive_offsets(tree: &Tree) -> Vec<(Depth, Range<usize>)> {
        let root_node = tree.root_node();
        let mut offsets = vec![];
        recursive_naive_offsets(&mut offsets, root_node, 0);
        offsets
    }

    // Basic version to compare an optimized version against. According to the tree-sitter
    // documentation, this is not efficient for large trees. But because it is the easiest
    // to reason about it is a good check for correctness.
    fn recursive_naive_offsets(
        collection: &mut Vec<(Depth, Range<usize>)>,
        node: Node<'_>,
        depth: usize,
    ) {
        // We can skip the root node
        if depth > 0 {
            collection.push((Depth(depth), node.byte_range()));
        }

        for child in node.children(&mut node.walk()) {
            recursive_naive_offsets(collection, child, depth + 1);
        }
    }
}

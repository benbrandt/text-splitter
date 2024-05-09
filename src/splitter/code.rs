#[cfg(test)]
mod tests {
    use std::{cmp::Ordering, ops::Range};

    use tree_sitter::{Node, Parser, Tree, TreeCursor};

    use crate::splitter::SemanticLevel;

    /// New type around a usize to capture the depth of a given code node.
    /// Custom type so that we can implement custom ordering, since we want to
    /// sort items of lower depth as higher priority.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Depth(usize);

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
            if self.cursor.goto_first_child()
                // There are sibling elements to grab because we are the deepest level
                || self.cursor.goto_next_sibling()
                // Go up and over to continue along the tree
                || (self.cursor.goto_parent() && self.cursor.goto_next_sibling())
            {
                Some((
                    Depth(self.cursor.depth() as usize),
                    self.cursor.node().byte_range(),
                ))
            } else {
                None
            }
        }
    }

    impl SemanticLevel for Depth {
        // const TRIM: Trim = Trim::PreserveIndentation;
        // fn offsets(text: &str) -> Vec<(Self, Range<usize>)> {
        //     let mut parser = Parser::new();
        //     parser
        //         .set_language(&tree_sitter_rust::language())
        //         .expect("Error loading Rust grammar");
        //     // The only reason the tree would be None is:
        //     // - No language was set (we do that)
        //     // - There was a timeout or cancellation option set (we don't)
        //     // - So it should be safe to unwrap here
        //     let tree = parser.parse(text, None).expect("Error parsing source code");

        //     CursorOffsets::new(tree.walk()).collect()
        // }
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

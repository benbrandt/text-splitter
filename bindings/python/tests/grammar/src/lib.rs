//! Expose a native test grammar without depending on a Python grammar wheel.

/// Return the static tree-sitter Rust language used by the threading tests.
#[no_mangle]
pub extern "C" fn text_splitter_test_language() -> *const std::ffi::c_void {
    // SAFETY: the grammar's generated entry point returns its static language.
    unsafe { tree_sitter_rust::LANGUAGE.into_raw()().cast() }
}

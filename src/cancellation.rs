use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

/// A token that can be used to cancel long-running operations.
///
/// This is a thread-safe cancellation token that can be cloned and shared
/// across threads. When cancelled, any operation checking this token should
/// stop processing and return early.
///
/// # Example
///
/// ```
/// use text_splitter::CancellationToken;
/// use std::thread;
/// use std::time::Duration;
///
/// let token = CancellationToken::new();
/// let token_clone = token.clone();
///
/// thread::spawn(move || {
///     // In another thread, cancel the operation
///     thread::sleep(Duration::from_millis(100));
///     token_clone.cancel();
/// });
///
/// // Check if cancelled before doing work
/// while !token.is_cancelled() {
///     // Do work...
///     # break; // For doctest
/// }
/// ```
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    inner: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Create a new cancellation token that is not cancelled.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Check if the token has been cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.inner.load(Ordering::Relaxed)
    }

    /// Cancel the token.
    ///
    /// This will cause any operation checking this token to stop processing.
    pub fn cancel(&self) {
        self.inner.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_token_not_cancelled_by_default() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn cancellation_token_can_be_cancelled() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn cancellation_token_clone_shares_state() {
        let token = CancellationToken::new();
        let token_clone = token.clone();

        assert!(!token.is_cancelled());
        assert!(!token_clone.is_cancelled());

        token.cancel();

        assert!(token.is_cancelled());
        assert!(token_clone.is_cancelled());
    }

    #[test]
    fn cancellation_token_thread_safety() {
        use std::thread;

        let token = CancellationToken::new();
        let token_clone = token.clone();

        thread::spawn(move || {
            thread::sleep(std::time::Duration::from_millis(10));
            token_clone.cancel();
        });

        // Wait for cancellation
        thread::sleep(std::time::Duration::from_millis(20));
        assert!(token.is_cancelled());
    }
}

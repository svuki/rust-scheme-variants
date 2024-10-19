# Rust Scheme Experiments

This repo contains multiple implementations of scheme interpreters that support various subsets of RSR7 scheme functionality. The primary purpose in implementing these was to learn rust and explore its memory management techniques.

The implementions can be found in `src/`. Each implementation is included as its own module.

Here are descriptions for each interpreter.

## src/basic.rs

This implements enough of scheme to define and evaluate the factorial functions.

Cloning is used heavily here to avoid references.

## src/basic_refs.rs

This is the same implementation as basic, but we try to remove cloning where possible.

## src/closures.rs

This implementation adds closures and mutable enironments. It introduces the use of Rc<RefCell<T>> to support multuple references to multiple environments.

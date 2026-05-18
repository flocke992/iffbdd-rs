# iffbdd-rs
A repository to minimize quadratic objectives under linear constraints using message-passing algorithms.
Especially the first-order solver achieves state-of-the-art speeds. It is an implementation of the dual coordinate descent algorithm.
This repo is based on my master project.

## Doc
use `cargo doc` to create a simple documentation.

## Example usage
Some examples are provied in the `example/` directory.

For larger problems, we encourage to use the helper struct `ProblemConfig` to setup the solver.

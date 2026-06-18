# iffbdd-rs
A repository to minimize quadratic objectives under linear constraints using message-passing algorithms.
The problem formulation is given by:

$$\text{minimize}\ Q(x)\\ \text{subject to linear constraints,}$$ 

where $Q(x)$ is a quadratic expression and $x\in\mathbb{R}^k$.

A first-order and a second-order solver are provided.
Especially the first-order solver achieves state-of-the-art speeds. It is an implementation of the dual coordinate descent algorithm.

This repo is based on my master project.

## Doc
use `cargo doc` to create a simple documentation.

## Example usage
Some examples are provied in the `example/` directory.

For larger problems, we encourage to use the helper struct `ProblemConfig` to setup the solver.

``` rust
    // Example setup
    let obj = Objective::L1;
    let c1 = ConstraintSense::Less(1.0, Box::from([1.0, 0.0, 0.5, -1.0, 1.0]));
    let c2 = ConstraintSense::Greater(1.0, Box::from([0.0, 5.0, 3.0, -2.0, 0.0]));
    let c3 = ConstraintSense::Equal(5.0, Box::from([1.0, 1.0, 0.0, 0.0, 1.0]));
    let c4 = ConstraintSense::Less(1.0, Box::from([-1.0, -0.5, 0.5, 1.5, 1.0]));
    let config = ProblemConfig {
        k: 5,
        solver: SolverType::Dcd, //First-order solver
        objective: obj,
        constraints: vec![c1, c2, c3, c4],
    };
    // init solver with constraints
    let mut s = init_solver_from_config(config);
    // solve objective under constraints
    let res = s.solve(obj);
    match res {
        Ok(sol) => {
            println!("Success");
            println!("{sol:.4?}");
        }
        Err(e) => {
            println!("No Success");
            println!("{e:?}");
        }
    }

```

## Performance
We repeat the experiment for the given dimensions $k$ & $n$ serveral times and plot the different timed runs as a violine plot.
### Half-Space Constraints
![./l2_ineq.pdf]
### Equality Constraints
![./l2_eq.pdf]

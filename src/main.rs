mod constraint_optimizer;
mod file_parser;
mod performance;

use constraint_optimizer::constr_optim::*;
use constraint_optimizer::helper::*;

use crate::performance::observe_single_solver;
use crate::performance::{generate_feasible_constraints, repeat_experiment, write_csv};

fn main() {
    println!("Solve ex3:");
    let maybe_solver = init_from_file("example/ex3.txt");
    match maybe_solver {
        Ok((obj, mut s)) => {
            println!("Solve with objective: {obj:?}");
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
        }
        Err(e) => {
            println!("There was a problem");
            println!("{e}");
        }
    }
    println!("Solve ex4:");
    let maybe_solver = init_from_file("example/ex4.txt");
    match maybe_solver {
        Ok((obj, mut s)) => {
            println!("Solve with objective: {obj:?}");
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
        }
        Err(e) => {
            println!("There was a problem");
            println!("{e}");
        }
    }

    let obj = Objective::L1;
    let c1 = ConstraintSense::Less(1.0, Box::from([1.0, 0.0, 0.5, -1.0, 1.0]));
    let c2 = ConstraintSense::Greater(1.0, Box::from([0.0, 5.0, 3.0, -2.0, 0.0]));
    let c3 = ConstraintSense::Equal(5.0, Box::from([1.0, 1.0, 0.0, 0.0, 1.0]));
    let c4 = ConstraintSense::Less(1.0, Box::from([-1.0, -0.5, 0.5, 1.5, 1.0]));
    let config = ProblemConfig {
        k: 5,
        solver: SolverType::Dcd,
        objective: obj,
        constraints: vec![c1, c2, c3, c4],
    };
    let mut s = init_solver_from_config(config);
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

    let k = 10_usize;

    let constr = generate_feasible_constraints(k, k, SenseType::Less);
    println!("{constr:?}");
    let conf = ProblemConfig {
        solver: SolverType::Dcd,
        objective: Objective::L2,
        k,
        constraints: constr,
    };

    let mut s = init_solver_from_config(conf);
    match s.solve(Objective::L2) {
        Ok(x) => {
            println!("Success");
            println!("{x:.4?}");
        }
        Err(e) => {
            println!("No Success");
            println!("{e:?}");
        }
    }

    // let k_n = vec![
    //     (5, 1),
    //     (10, 2),
    //     (30, 6),
    //     (50, 10),
    //     (100, 20),
    //     (300, 60),
    //     (500, 100),
    // ];
    // let k_n = vec![
    //     (5, 5),
    //     (10, 10),
    //     (30, 30),
    //     (50, 50),
    //     (100, 100),
    //     (300, 300),
    //     (500, 500),
    // ];
    //
    // let solvers = vec![SolverType::Iffbdd, SolverType::Dcd];
    // let objectives = vec![Objective::L2];
    //
    // let repetitions = 80;
    // for s in &solvers {
    //     for o in &objectives {
    //         let res = observe_single_solver(&k_n, SenseType::Less, *o, repetitions, *s);
    //         // let res = observe_single_solver(&k_n, SenseType::Equal, *o, repetitions, *s);
    //         let names = k_n.iter().map(|(k, _n)| k.to_string()).collect();
    //         let _ = write_csv(&format!("single_{s}_{o}.csv",), names, res);
    //     }
    // }
}

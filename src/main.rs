mod constraint_optimizer;
mod file_parser;

use constraint_optimizer::constr_optim::*;
use constraint_optimizer::helper::*;

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
}

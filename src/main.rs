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
}

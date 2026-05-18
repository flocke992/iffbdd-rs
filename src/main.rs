mod constraint_optimizer;
mod file_parser;

use constraint_optimizer::constr_optim::*;
use constraint_optimizer::helper::*;

fn main() {
    let maybe_solver = init_from_file("setup.txt");
    match maybe_solver {
        Ok((obj, mut s)) => {
            println!("Solve with objective: {obj:?}");
            let res = s.solve(obj);
            match res {
                Ok(sol) => {
                    println!("Success");
                    println!("{sol:?}");
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

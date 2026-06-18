use csv::Writer;
use std::{error::Error, fs::File, time::Instant};

use crate::constraint_optimizer::{
    constr_optim::{ConstraintSense, Objective, SolveError},
    helper::{ProblemConfig, SenseType, SolverType, init_solver_from_config},
};

pub fn write_csv(
    csv_name: &str,
    column_headers: Vec<String>,
    data: Vec<Vec<f64>>,
) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_writer(File::create(csv_name)?);
    wtr.write_record(column_headers)?;

    for row in data {
        wtr.write_record(row.into_iter().map(|p| p.to_string()))?;
    }
    Ok(())
}

fn generate_feasible_mappings(k: usize, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    //arbitrary number range
    let a: Vec<f64> = (0..k * n)
        .into_iter()
        .map(|_| rand::random_range(-10.0..10.0))
        .collect();
    let x: Vec<f64> = (0..k)
        .into_iter()
        .map(|_| rand::random_range(-10.0..10.0))
        .collect();
    let mut b = vec![0.0; n];

    // manual matrix-vec multiplication
    for (i, a_i) in a.chunks_exact(k).enumerate() {
        let b_i = x
            .iter()
            .zip(a_i)
            .fold(0.0, |acc, (x_i, a_ii)| acc + x_i * a_ii);
        b[i] = b_i;
    }
    (a, x, b)
}

pub fn generate_feasible_constraints(
    k: usize,
    n: usize,
    sense_type: SenseType,
) -> Vec<ConstraintSense> {
    let (a, _, b) = generate_feasible_mappings(k, n);
    match sense_type {
        SenseType::Interval => panic!("don't use this"),
        SenseType::Less => {
            return a
                .chunks_exact(k)
                .zip(b)
                .map(|(a_i, b_i)| ConstraintSense::Less(b_i, Box::from(a_i)))
                .collect();
        }
        SenseType::Equal => {
            return a
                .chunks_exact(k)
                .zip(b)
                .map(|(a_i, b_i)| ConstraintSense::Equal(b_i, Box::from(a_i)))
                .collect();
        }
        SenseType::Greater => {
            return a
                .chunks_exact(k)
                .zip(b)
                .map(|(a_i, b_i)| ConstraintSense::Greater(b_i, Box::from(a_i)))
                .collect();
        }
    }
}

fn check_time(conf: ProblemConfig) -> Option<f64> {
    let obj = conf.objective;
    let mut s = init_solver_from_config(conf);
    let start_time = Instant::now();
    match s.solve(obj) {
        Ok(_) => Some(start_time.elapsed().as_secs_f64()),
        Err(e) => match e {
            SolveError::NoConvergence => None,
            SolveError::NotFeasible(viol, _) => {
                println!("Violated with: {viol:.3}");
                None
            }
        },
    }
}

pub fn observe_single_solver(
    k_n: &[(usize, usize)],
    sense_type: SenseType,
    obj: Objective,
    repetitions: usize,
    solver: SolverType,
) -> Vec<Vec<f64>> {
    let mut res = Vec::new();
    for _i in 0..repetitions {
        let mut inner_res = Vec::new();
        for (k, n) in k_n {
            let constr = generate_feasible_constraints(*k, *n, sense_type);
            let conf = ProblemConfig {
                solver: solver,
                objective: obj,
                k: *k,
                constraints: constr,
            };
            if let Some(time) = check_time(conf) {
                inner_res.push(time);
            } else {
                println!("Couldn't solve, pushing -1");
                inner_res.push(-1.0);
            }
        }
        res.push(inner_res);
    }
    res
}

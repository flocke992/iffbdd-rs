use crate::constraint_optimizer::constr_optim::{ConstraintSense, Objective};
use crate::constraint_optimizer::helper::{ProblemConfig, SenseType, SolverType};
use std::fs;
use std::path::Path;
use std::str::FromStr;

#[derive(Debug)]
pub enum ParseError {
    Io(std::io::Error),
    MissingSolver,
    InvalidSolver(String),
    MissingObjective,
    InvalidObjective(String),
    InvalidSense(String),
    InvalidFloat(String),
    MalformedConstraint(String),
    InconsistentDimension {
        expected: usize,
        got: usize,
        line: usize,
    },
    EmptyFile,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::Io(e) => write!(f, "IO error: {e}"),
            ParseError::MissingSolver => write!(f, "Missing 'solver:' declaration"),
            ParseError::InvalidSolver(s) => {
                write!(f, "Unknown solver '{s}', expected DCD or IFFBDD")
            }
            ParseError::MissingObjective => write!(f, "Missing 'objective:' declaration"),
            ParseError::InvalidObjective(s) => {
                write!(f, "Unknown objective '{s}', expected L1 or L2")
            }
            ParseError::InvalidSense(s) => write!(
                f,
                "Unknown sense '{s}', expected Less/Greater/Equal/Interval"
            ),
            ParseError::InvalidFloat(s) => write!(f, "Could not parse float: '{s}'"),
            ParseError::MalformedConstraint(s) => write!(f, "Malformed constraint line: '{s}'"),
            ParseError::InconsistentDimension {
                expected,
                got,
                line,
            } => write!(
                f,
                "Line {line}: expected {expected} coefficients, got {got}"
            ),
            ParseError::EmptyFile => write!(f, "File is empty or has no content"),
        }
    }
}

impl From<std::io::Error> for ParseError {
    fn from(e: std::io::Error) -> Self {
        ParseError::Io(e)
    }
}

impl std::error::Error for ParseError {}

// --- Helpers ---

fn parse_floats(s: &str, context: &str) -> Result<Box<[f64]>, ParseError> {
    s.split_whitespace()
        .map(|tok| {
            f64::from_str(tok)
                .map_err(|_| ParseError::InvalidFloat(format!("{tok} (in '{context}')")))
        })
        .collect()
}

fn parse_sense(s: &str) -> Result<SenseType, ParseError> {
    match s.trim() {
        "Less" => Ok(SenseType::Less),
        "Greater" => Ok(SenseType::Greater),
        "Equal" => Ok(SenseType::Equal),
        "Interval" => Ok(SenseType::Interval),
        other => Err(ParseError::InvalidSense(other.to_string())),
    }
}

pub fn parse_problem_file(path: impl AsRef<Path>) -> Result<ProblemConfig, ParseError> {
    let content = fs::read_to_string(path)?;
    parse_problem_str(&content)
}

pub fn parse_problem_str(content: &str) -> Result<ProblemConfig, ParseError> {
    let mut objective: Option<Objective> = None;
    let mut solver: Option<SolverType> = None;
    let mut constraints: Vec<ConstraintSense> = Vec::new();
    let mut k: Option<usize> = None;

    if content.is_empty() {
        return Err(ParseError::EmptyFile);
    }

    for (line_num, raw_line) in content.lines().enumerate() {
        // Strip comments and skip blank lines
        let line = match raw_line.split('#').next() {
            Some(l) => l.trim(),
            None => continue,
        };
        if line.is_empty() {
            continue;
        }

        if let Some(value) = line.strip_prefix("solver:") {
            let token = value.trim();
            solver = Some(match token {
                "DCD" => SolverType::Dcd,
                "IFFBDD" => SolverType::Iffbdd,
                other => return Err(ParseError::InvalidSolver(other.to_string())),
            });
        } else if let Some(value) = line.strip_prefix("objective:") {
            let token = value.trim();
            objective = Some(match token {
                "L1" => Objective::L1,
                "L2" => Objective::L2,
                other => return Err(ParseError::InvalidObjective(other.to_string())),
            });
        } else if let Some(value) = line.strip_prefix("constraint:") {
            // Split into three pipe-separated sections
            let parts: Vec<&str> = value.splitn(3, '|').collect();
            if parts.len() != 3 {
                return Err(ParseError::MalformedConstraint(line.to_string()));
            }

            let coefficients = parse_floats(parts[0], line)?;
            let bounds = parse_floats(parts[1], line)?;
            let sense = parse_sense(parts[2])?;

            // Validate / infer dimension k
            let coeff_len = coefficients.len();
            match k {
                None => k = Some(coeff_len),
                Some(expected) if coeff_len != expected => {
                    return Err(ParseError::InconsistentDimension {
                        expected,
                        got: coeff_len,
                        line: line_num + 1,
                    });
                }
                _ => {}
            }

            match sense {
                SenseType::Less => {
                    if bounds.len() != 1 {
                        return Err(ParseError::MalformedConstraint(line.to_string()));
                    }
                    constraints.push(ConstraintSense::Less(bounds[0], coefficients));
                }
                SenseType::Equal => {
                    if bounds.len() != 1 {
                        return Err(ParseError::MalformedConstraint(line.to_string()));
                    }
                    constraints.push(ConstraintSense::Equal(bounds[0], coefficients));
                }
                SenseType::Greater => {
                    if bounds.len() != 1 {
                        return Err(ParseError::MalformedConstraint(line.to_string()));
                    }
                    constraints.push(ConstraintSense::Greater(bounds[0], coefficients));
                }
                SenseType::Interval => {
                    if bounds.len() != 2 {
                        return Err(ParseError::MalformedConstraint(line.to_string()));
                    }
                    constraints.push(ConstraintSense::Interval(
                        (bounds[0], bounds[1]),
                        coefficients,
                    ));
                }
            }
        } else {
            println!("did run in an unhandled case, continue...");
        }
    }

    let solver = solver.ok_or(ParseError::MissingSolver)?;
    let objective = objective.ok_or(ParseError::MissingObjective)?;
    let k = k.unwrap_or(0);

    Ok(ProblemConfig {
        solver,
        objective,
        k,
        constraints,
    })
}

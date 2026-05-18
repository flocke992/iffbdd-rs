//! # Constraint Optimizer
//!
//! This module provides implementations of iterative algorithms for solving quadratic programming problems
//! with linear constraints. The main goal is to find a vector `x` that minimizes the L1 or L2 norm
//! subject to a set of linear constraints of the form `a_i^T x (<=, >=, ==, [b0, b1]) b`.
//!
//! ## Main Components
//!
//! ### Module [constr_optim](mod@constr_optim)
//! This module implements the core logic of the solvers.
//!
//! - **ConstraintSense**: Structure representing a single constraint with its coefficients and sense.
//! - **Solver Trait**: Abstract interface for constraint solvers, supporting constraint addition, solving, and constraint checking.
//! - **SecondOrderSolver**: Implements a second-order iterative filtering algorithm for constraint satisfaction. Usually requires less iterations but one iteration is more expensive.
//! - **FirstOrderSolver**: Implements a first-order iterative algorithm for constraint satisfaction. Usually requires more iterations but one iteration is less expensive.
//!
//! The algorithms are designed for efficiency and scalability, leveraging BLAS routines for linear algebra operations.
//! ### Module [helper](mod@helper)
//! This module provides helper methods for ease of implementation.
//!
//! - **SenseType**: Enumeration to specify the type of constraints (Less, Greater, Equal, Interval).
//! - **SolverType**: Enumeration to specify the type solver
//!
//! ## Example Usage
//!
//! ```rust
//!
//! // Suppose we want to solve: minimize ||x||^2 subject to x[0] + x[1] <= 1, x[0] - x[1] >= 0
//! let k = 2;
//! let a1 = [1.0, 1.0];
//! let a2 = [1.0, -1.0];
//! let b1 = 1.0;
//! let b2 = 0.0;
//!
//! // SecondOrderSolver example
//! let mut solver = SecondOrderSolver::new(k);
//! solver.add_constraint(ConstraintSense::Less(b1, &a1));
//! solver.add_constraint(ConstraintSense::Greater(b2, &a2));
//! if let Ok(x_hat) = solver.solve() {
//!     println!("SecondOrderSolver solution: {:?}", x_hat);
//!     println!("Max constraint violation: {}", solver.check_constraints(&x_hat));
//! }
//! ```

pub mod constr_optim {
    extern crate openblas_src;
    use blas::{daxpy, dcopy, ddot, dgemv, dger, dscal};
    use std::error::Error;
    use std::fmt;

    /// Enum to express constraint, must contain the wanted border ``b`` / ``b0,b1`` and the linear mapping ``a_i`` as a slice.
    /// The caller ensures correct dimensions.
    #[derive(Clone, Debug)]
    pub enum ConstraintSense {
        Less(f64, Box<[f64]>),
        Greater(f64, Box<[f64]>),
        Equal(f64, Box<[f64]>),
        Interval((f64, f64), Box<[f64]>),
    }

    /// Enum to express Solver errors. The variants give info why the solver failed.
    #[derive(Debug)]
    pub enum SolveError {
        /// The solver did not converge in the given number of ``max_iter``.
        NoConvergence,
        /// The solver did converge but the maximum violation [`Solver::check_constraints()`] was larger than ``max_violation``.
        /// Maybe one can still continue with the produced candidate ``x_hat``.
        NotFeasible(Vec<f64>),
    }
    #[doc(hidden)]
    impl fmt::Display for SolveError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                SolveError::NoConvergence => write!(
                    f,
                    "The algorithm did not converge in the maximum allowed iterations. Maybe try increasing the number of iterations."
                ),
                SolveError::NotFeasible(_) => write!(
                    f,
                    "The algorithm did converge, but the maximum violation is outside the allowed range. Maybe try increasing the tolerance."
                ),
            }
        }
    }
    #[doc(hidden)]
    impl Error for SolveError {}

    /// Enum to express norm objective to minimize.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Objective {
        /// Minimize `||x||_1`
        L1,
        /// Minimize `||x||_2`
        L2,
    }

    /// Helperfunction to do an element-wise vector-vector product. ``y = a(y dot x)``.
    #[inline(always)]
    fn hadamard_product(y: &mut [f64], x: &[f64], a: f64) {
        y.iter_mut().zip(x).for_each(|(y_i, x_i)| *y_i *= a * x_i);
    }

    /// The trait with its methods which a solver must implement. Call these to set up your problem.
    pub trait Solver {
        /// Adds a constraint to the instantiated solver.
        /// Must pass in an ConstraintSense variant with its border ``b`` & linear mapping ``a_i``.
        fn add_constraint(&mut self, constraint: ConstraintSense);
        /// Minimizes an objective subject to the given set of linear constraints.
        /// Returns Result with custom error for further problem investigations.
        fn solve(&mut self, objective: Objective) -> Result<Vec<f64>, SolveError>;
        /// Checks the maximum violation of the set of constraints of a candidate solution ``x_hat``.
        fn check_constraints(&self, x_hat: &[f64]) -> f64;
    }

    /// Enum to express constraints and store internal variables for [`Constraint`](struct@Constraint)
    pub enum SenseLayout {
        Less(f64, f64),
        Greater(f64, f64),
        Equal(f64, f64),
        Interval((f64, f64), (f64, f64)),
    }

    /// Implements the logic of the [`Objective`](enum@Objective)
    struct Prior {
        diag_cov: Vec<f64>,
        objective: Objective,
        k: usize,
    }

    impl Prior {
        fn new(k: usize, objective: Objective) -> Self {
            Prior {
                diag_cov: vec![1.0; k],
                objective,
                k,
            }
        }

        fn calculate_primal_posterior(&self, x_hat: &mut [f64]) {
            match self.objective {
                Objective::L2 => unsafe {
                    dscal(self.k as i32, -1.0, x_hat, 1);
                },
                _ => {
                    hadamard_product(x_hat, &self.diag_cov, -1.0);
                }
            }
        }

        fn update_prior(&mut self, primal_posterior: &[f64]) {
            match self.objective {
                Objective::L2 => (),
                Objective::L1 => {
                    self.diag_cov
                        .iter_mut()
                        .zip(primal_posterior)
                        .for_each(|(c_i, x_i)| *c_i = x_i.abs());
                }
            }
        }
    }

    /// Single constraint, which needs to be satisfied. Implements [`Self::new()`] which is called by [`Solver::add_constraint()`] to add a new constraint.
    /// [`Self::forward_filter()`], [`Self::backward_decide()`] implement the forward filter & backward decide gaussian message passing logic. They are called in the [`Solver::solve()`] method.
    struct Constraint {
        /// Constraint sense configuration and internal parameters
        sense: SenseLayout,
        /// Backward mean message
        mb_y: f64,
        /// Backward variance message
        vb_y: f64,
        /// linear mapping ``a_i``
        pub a_i: Vec<f64>,
        /// dimension of ``x_hat`` and ``a_i``
        k: i32,
        /// stored variable to reuse in [`Self::backward_decide()`]
        variance_norm_store: f64,
        /// stored variable to reuse in [`Self::backward_decide()`]
        mean_store: f64,
        /// stored vector to reuse in [`Self::backward_decide()`]
        variance_store: Vec<f64>,
    }
    impl Constraint {
        fn new(constraint_sense: ConstraintSense) -> Self {
            let (a_i, sense) = match constraint_sense {
                ConstraintSense::Less(b, a_i) => (a_i.to_vec(), SenseLayout::Less(b, 0.0)),
                ConstraintSense::Greater(b, a_i) => (a_i.to_vec(), SenseLayout::Greater(b, 0.0)),
                ConstraintSense::Equal(b, a_i) => (a_i.to_vec(), SenseLayout::Equal(b, 0.0)),
                ConstraintSense::Interval((b0, b1), a_i) => {
                    (a_i.to_vec(), SenseLayout::Interval((b0, b1), (0.0, 0.0)))
                }
            };
            let mb_y = 0.0;
            let vb_y = 1.0;
            let k = a_i.len() as i32;
            let variance_norm_store = 1.0;
            let mean_store = 0.0;
            let variance_store = vec![0.0; k as usize];
            Constraint {
                sense,
                mb_y,
                vb_y,
                a_i,
                k,
                variance_norm_store,
                mean_store,
                variance_store,
            }
        }

        /// Updates forward mean ``m_x`` and covariance message ``v_x`` based on determined backward message ``mb_y, vb_y`` in the backward decide pass.
        /// This is the standard Kalmann forward filter step.
        fn forward_filter(&mut self, m_x: &mut [f64], v_x: &mut [f64], temp_vec: &mut [f64]) {
            let mut g;
            unsafe {
                // temp_vec = v_x @ a_i^T
                dgemv(
                    b'N',
                    self.k,
                    self.k,
                    1.0,
                    v_x,
                    self.k,
                    &self.a_i[..],
                    1,
                    0.0,
                    temp_vec,
                    1,
                );

                // g = a_i @ v_x @ a_i^T
                g = ddot(self.k, temp_vec, 1, &self.a_i[..], 1);
                //store new a_i @ m_x
                self.mean_store = ddot(self.k, m_x, 1, &self.a_i[..], 1);
                //variance_store = v_x @ a_i^T
                dcopy(self.k, temp_vec, 1, &mut self.variance_store, 1);
            }

            self.variance_norm_store = g;

            if self.vb_y < 1.0e14 {
                g = 1.0 / (self.vb_y + g);

                unsafe {
                    //update m_x
                    //m_x = m_x + v_x @ a_i^T * g * (mb_y - a_i @ m_x)
                    daxpy(
                        self.k,
                        g * (self.mb_y - self.mean_store),
                        temp_vec,
                        1,
                        m_x,
                        1,
                    );
                    //update v_x
                    //v_x = v_x - g * (v_x @ a_i^T) @ (v_x @ a_i^T)^T
                    dger(self.k, self.k, -g, temp_vec, 1, temp_vec, 1, v_x, self.k);
                }
            }
        }

        /// Updates fixed dual backward message ``x_hat`` and backward message ``mb_y, vb_y`` based on stored values from [`Self::forward_filter()`] and ``x_hat``
        fn backward_decide(&mut self, x_hat: &mut [f64]) {
            let m_y;
            unsafe {
                // m_y = a_i @ m_x - a_i @ v_x @ x_hat
                m_y = self.mean_store - ddot(self.k, &self.variance_store, 1, x_hat, 1);
            }
            // updates mb_y, vb_y & x_hat according to update table
            match self.sense {
                SenseLayout::Less(b, ref mut g0) => {
                    if m_y <= b {
                        *g0 = g0.max((b - m_y) / 2.0);
                        self.vb_y = f64::INFINITY;
                    } else {
                        let z_hat = (m_y - b) / self.variance_norm_store;
                        self.vb_y = *g0 / z_hat;
                        self.mb_y = b - *g0;
                        unsafe {
                            daxpy(self.k, z_hat, &self.a_i[..], 1, x_hat, 1);
                        }
                    }
                }
                SenseLayout::Greater(b, ref mut g0) => {
                    if m_y >= b {
                        *g0 = g0.max((b - m_y) / 2.0);
                        self.vb_y = f64::INFINITY;
                    } else {
                        let z_hat = (m_y - b) / self.variance_norm_store;
                        self.vb_y = *g0 / -z_hat;
                        self.mb_y = *g0 + b;
                        unsafe {
                            daxpy(self.k, z_hat, &self.a_i[..], 1, x_hat, 1);
                        }
                    }
                }
                SenseLayout::Equal(b, ref mut _g0) => {
                    let z_hat = (m_y - b) / self.variance_norm_store;
                    self.vb_y = 0.0;
                    self.mb_y = b;
                    unsafe {
                        daxpy(self.k, z_hat, &self.a_i[..], 1, x_hat, 1);
                    }
                }
                SenseLayout::Interval((b0, b1), (ref mut g0, ref mut g1)) => {
                    if m_y < b0 {
                        let z_hat = (m_y - b0) / self.variance_norm_store;
                        self.vb_y = -*g0 / z_hat;
                        self.mb_y = *g0 + b0;
                        unsafe {
                            daxpy(self.k, z_hat, &self.a_i[..], 1, x_hat, 1);
                        }
                        *g1 = g1.max((b1 - m_y) / 2.0);
                    } else if m_y > b1 {
                        let z_hat = (m_y - b1) / self.variance_norm_store;
                        self.vb_y = *g1 / z_hat;
                        self.mb_y = b1 - *g1;
                        unsafe {
                            daxpy(self.k, z_hat, &self.a_i[..], 1, x_hat, 1);
                        }
                        *g0 = g0.max((m_y - b0) / 2.0);
                    } else {
                        *g0 = g0.max((m_y - b0) / 2.0);
                        *g1 = g1.max((b1 - m_y) / 2.0);
                        self.vb_y = f64::INFINITY;
                    }
                }
            }
        }

        /// checks violation of constraint
        fn check_constraint(&self, x_hat: &[f64]) -> f64 {
            let y_hat;
            unsafe {
                y_hat = ddot(self.k, &self.a_i[..], 1, x_hat, 1);
            }
            match self.sense {
                SenseLayout::Equal(b, _) => (y_hat - b).abs(),
                SenseLayout::Greater(b, _) => (b - y_hat).max(0.0),
                SenseLayout::Less(b, _) => (y_hat - b).max(0.0),
                SenseLayout::Interval((b0, b1), _) => (b0 - y_hat).max(y_hat - b1).max(0.0),
            }
        }
    }

    /// Second-order solver which impelments the [`Solver`] trait.
    pub struct SecondOrderSolver {
        /// Vector of stored [`Constraint`]s
        constraints: Vec<Constraint>,
        /// Dimension of ``x_hat``.
        k: usize,
        /// Forward mean message.
        m_x: Vec<f64>,
        /// Forward covariance message.
        v_x: Vec<f64>,
        /// Allocated vector to use for intermediate computations.
        temp_vec: Vec<f64>,
        /// old norm of candidate solution to check for convergence.
        old_x_hat_norm: f64,
        /// old norm of internal parameters to check for convergence.
        old_gammas_norm: f64,
        //hyperparams
        /// Convergence tolerance between ``|old_value - new_value|``.
        convergence_tol: f64,
        /// Maximum number of iterations for algorithm until convergence must be met.
        max_iter: usize,
        /// Maximum allowed constraint violation before declaring infeasibility.
        max_viol: f64,
    }

    impl SecondOrderSolver {
        /// Call this method to instantiate a new second-order solver.
        pub fn new(k: usize) -> Self {
            let m_x = vec![0.0; k];
            let v_x = vec![0.0; k * k];
            let temp_vec = vec![0.0; k];
            Self {
                constraints: Vec::new(),
                k,
                m_x,
                v_x,
                temp_vec,
                convergence_tol: 1e-12,
                old_x_hat_norm: f64::MAX,
                old_gammas_norm: f64::MAX,
                max_iter: 2_000_000,
                max_viol: 1e-6,
            }
        }

        /// Internal method to check if ``x_hat`` has converged.
        fn check_convergence_candidate(&mut self) -> bool {
            let x_hat_norm = self.m_x.iter().fold(0.0, |acc, x_i| acc + x_i.powi(2));
            let res = (x_hat_norm - self.old_x_hat_norm).abs() < self.convergence_tol;
            self.old_x_hat_norm = x_hat_norm;
            res
        }

        /// Internal method to check if internal variables have converged.
        fn check_convergence_gammas(&mut self) -> bool {
            let gammas_norm = self.constraints.iter().fold(0.0, |acc, c| {
                acc + match c.sense {
                    SenseLayout::Less(_, g0)
                    | SenseLayout::Greater(_, g0)
                    | SenseLayout::Equal(_, g0) => g0.powi(2),
                    SenseLayout::Interval(_, (g0, g1)) => g0.powi(2) + g1.powi(2),
                }
            });
            let res = (gammas_norm - self.old_gammas_norm).abs() < self.convergence_tol;
            self.old_gammas_norm = gammas_norm;
            res
        }

        /// Internal helper method to reset gaussian messages.
        fn reset_forward_message(&mut self, diag_prior: &[f64]) {
            self.m_x.fill(0.0);
            self.v_x.fill(0.0);
            //fill diagonal elements
            //(0..self.k).for_each(|i| self.v_x[i * self.k + i] = 1.0);
            (0..self.k).for_each(|i| self.v_x[i * self.k + i] = diag_prior[i]);
        }
    }

    impl Solver for SecondOrderSolver {
        fn add_constraint(&mut self, constraint: ConstraintSense) {
            let constr = Constraint::new(constraint);
            self.constraints.push(constr);
        }

        fn solve(&mut self, objective: Objective) -> Result<Vec<f64>, SolveError> {
            let mut prior = Prior::new(self.k, objective);
            for _i in 0..self.max_iter {
                self.reset_forward_message(&prior.diag_cov);
                //forward Kalmann filter
                self.constraints.iter_mut().for_each(|c| {
                    c.forward_filter(&mut self.m_x, &mut self.v_x, &mut self.temp_vec);
                });

                //use m_x as x_hat
                self.m_x.fill(0.0);
                //backward dual decide
                self.constraints
                    .iter_mut()
                    .rev()
                    .for_each(|c| c.backward_decide(&mut self.m_x));

                //determine posterior primal and update prior based on that posterior
                prior.calculate_primal_posterior(&mut self.m_x);
                prior.update_prior(&self.m_x);

                //check for convergence
                if self.check_convergence_candidate() && self.check_convergence_gammas() {
                    if self.max_viol >= self.check_constraints(&self.m_x) {
                        return Ok(self.m_x.clone());
                    } else {
                        // Too much violation but convergence
                        return Err(SolveError::NotFeasible(self.m_x.clone()));
                    }
                }
            }
            // No convergence
            Err(SolveError::NoConvergence)
        }

        fn check_constraints(&self, x_hat: &[f64]) -> f64 {
            self.constraints
                .iter()
                .fold(0.0, |acc, c| acc.max(c.check_constraint(x_hat)))
        }
    }

    /// Enum to express constraints and store internal variables for [`LimitConstraint`].
    enum LimitSenseLayout {
        Less(f64),
        Greater(f64),
        Equal(f64),
        Interval((f64, f64)),
    }

    /// Single constraint, which needs to be satisfied. Implements [`Self::new()`] which is called by [`Solver::add_constraint()`] to add a new constraint.
    /// [`Self::forward_filter()`], [`Self::backward_decide()`] implement the forward filter & backward decide gaussian message passing logic. They are called in the [`Solver::solve()`] method.
    struct LimitConstraint {
        /// Constraint sense configuration and internal parameters
        sense: LimitSenseLayout,
        /// linear mapping ``a_i``
        a_i: Vec<f64>,
        /// dimension of ``x_hat`` and ``a_i``
        k: i32,
        /// precomputed stored value ``1/(a_i @ a_i^T)`` to use in [`Self::backward_decide()`]
        percision_store: f64,
        /// allocated field to store forward and backward messages respectively
        message_store: f64,
        /// allocated vec to store ``V_x @ a_i^T``
        projected_variance: Vec<f64>,
    }

    impl LimitConstraint {
        pub fn new(constraint_sense: ConstraintSense) -> Self {
            let (a_i, sense) = match constraint_sense {
                ConstraintSense::Less(b, a_i) => (a_i.to_vec(), LimitSenseLayout::Less(b)),
                ConstraintSense::Greater(b, a_i) => (a_i.to_vec(), LimitSenseLayout::Greater(b)),
                ConstraintSense::Equal(b, a_i) => (a_i.to_vec(), LimitSenseLayout::Equal(b)),
                ConstraintSense::Interval((b0, b1), a_i) => {
                    (a_i.to_vec(), LimitSenseLayout::Interval((b0, b1)))
                }
            };
            let k = a_i.len() as i32;
            let message_store = 1.0;
            LimitConstraint {
                sense,
                a_i,
                k,
                percision_store: 0.0,
                message_store,
                projected_variance: vec![0.0; k as usize],
            }
        }

        /// Updates forward mean ``m_x`` based on determined fixed dual variable ``x_hat`` from the previous [`Self::backward_decide()`]
        fn forward_filter(&mut self, m_x: &mut [f64], v_x: &[f64]) {
            let z_hat = self.message_store;
            unsafe {
                //store a_i @ v_x
                dcopy(self.k, &self.a_i, 1, &mut self.projected_variance, 1);
                hadamard_product(&mut self.projected_variance, v_x, 1.0);
                //store a_i @ v_x @ a_i^T
                self.percision_store =
                    ddot(self.k, &self.a_i, 1, &self.projected_variance, 1).recip();
                //store a_i @ m_x for backward decide
                self.message_store = ddot(self.k, m_x, 1, &self.a_i, 1);
            }
            if z_hat != 0.0 {
                //update m_x if previous decided variable z_hat is nonzero.
                //m_x = m_x - a_i^T * z_hat
                unsafe {
                    daxpy(self.k, -z_hat, &self.projected_variance, 1, m_x, 1);
                }
            }
        }

        /// Updates fixed dual backward message ``x_hat`` and stored message ``z_hat`` based on stored values from [`Self::forward_filter()`] and ``x_hat``
        fn backward_decide(&mut self, x_hat: &mut [f64]) {
            let mf_y;
            unsafe {
                // mf_y = a_i @ m_x - a_i @ x_hat
                mf_y = self.message_store - ddot(self.k, &self.projected_variance, 1, x_hat, 1);
            }

            //update z_hat according to table
            match self.sense {
                LimitSenseLayout::Less(b) => {
                    self.message_store = ((mf_y - b) * self.percision_store).max(0.0);
                }
                LimitSenseLayout::Greater(b) => {
                    self.message_store = ((mf_y - b) * self.percision_store).min(0.0);
                }
                LimitSenseLayout::Equal(b) => {
                    self.message_store = (mf_y - b) * self.percision_store;
                }
                LimitSenseLayout::Interval((b0, b1)) => {
                    if mf_y < b0 {
                        self.message_store = (mf_y - b0) * self.percision_store;
                    } else if mf_y > b1 {
                        self.message_store = (mf_y - b1) * self.percision_store;
                    } else {
                        self.message_store = 0.0;
                    }
                }
            }

            if self.message_store != 0.0 {
                // update x_hat = x_hat + a_i^T * z_hat if necessary
                unsafe {
                    daxpy(self.k, self.message_store, &self.a_i, 1, x_hat, 1);
                }
            }
        }

        fn check_constraint(&self, x_hat: &[f64]) -> f64 {
            let y_hat;
            unsafe {
                y_hat = ddot(self.k, &self.a_i[..], 1, x_hat, 1);
            }
            match self.sense {
                LimitSenseLayout::Less(b) => (y_hat - b).max(0.0),
                LimitSenseLayout::Greater(b) => (b - y_hat).max(0.0),
                LimitSenseLayout::Equal(b) => (y_hat - b).abs(),
                LimitSenseLayout::Interval((b0, b1)) => (b0 - y_hat).max(y_hat - b1).max(0.0),
            }
        }
    }

    /// First-order solver which impelments the [`Solver`] trait.
    pub struct FirstOrderSolver {
        /// Vector of [`Constraint`]s.
        constraints: Vec<LimitConstraint>,
        /// Dimension of ``x_hat``.
        k: usize,
        /// Forward mean message.
        m_x: Vec<f64>,
        /// Old norm of candidate solution to check for convergence.
        old_x_hat_norm: f64,
        //hyperparams
        /// Convergence tolerance between ``|old_value - new_value|``.
        convergence_tol: f64,
        /// Maximum number of iterations for algorithm until convergence must be met.
        max_iter: usize,
        /// Maximum allowed constraint violation before declaring infeasibility.
        max_viol: f64,
    }

    impl FirstOrderSolver {
        /// Call this method to instantiate a new first-order solver.
        pub fn new(k: usize) -> Self {
            let m_x = vec![0.0; k];
            Self {
                constraints: Vec::new(),
                k,
                m_x,
                convergence_tol: 1e-12,
                old_x_hat_norm: f64::MAX,
                max_iter: 5_000_000,
                max_viol: 1e-8,
            }
        }

        /// Internal method to check if (`x_hat`) has converged.
        fn check_convergence_candidate(&mut self) -> bool {
            let x_hat_norm = self.m_x.iter().fold(0.0, |acc, x_i| acc + x_i.powi(2));
            let res = (x_hat_norm - self.old_x_hat_norm).abs() < self.convergence_tol;
            self.old_x_hat_norm = x_hat_norm;
            res
        }
    }

    impl Solver for FirstOrderSolver {
        fn add_constraint(&mut self, constraint: ConstraintSense) {
            let constr = LimitConstraint::new(constraint);
            self.constraints.push(constr);
        }

        fn solve(&mut self, objective: Objective) -> Result<Vec<f64>, SolveError> {
            let mut prior = Prior::new(self.k, objective);
            for _i in 0..self.max_iter {
                self.m_x.fill(0.0);
                //simplified forward Kalamann filter
                self.constraints
                    .iter_mut()
                    .for_each(|c| c.forward_filter(&mut self.m_x, &prior.diag_cov));

                //use m_x as x_hat
                self.m_x.fill(0.0);
                //simplified backward dual decide
                self.constraints
                    .iter_mut()
                    .rev()
                    .for_each(|c| c.backward_decide(&mut self.m_x));

                prior.calculate_primal_posterior(&mut self.m_x);
                prior.update_prior(&self.m_x);

                if self.check_convergence_candidate() {
                    if self.max_viol >= self.check_constraints(&self.m_x) {
                        return Ok(self.m_x.clone());
                    } else {
                        // Too much violation, but convergence
                        return Err(SolveError::NotFeasible(self.m_x.clone()));
                    }
                }
            }
            // No convergence
            Err(SolveError::NoConvergence)
        }

        fn check_constraints(&self, x_hat: &[f64]) -> f64 {
            //infinity norm of violated border per constraint
            self.constraints
                .iter()
                .fold(0.0, |acc, c| acc.max(c.check_constraint(x_hat)))
        }
    }
}

pub mod helper {
    use crate::ConstraintSense;
    use crate::Solver;
    use crate::constraint_optimizer::constr_optim::{
        FirstOrderSolver, Objective, SecondOrderSolver,
    };
    use crate::file_parser::{ParseError, parse_problem_file};
    use std::path::Path;
    /// What type of
    #[derive(Debug, Clone, PartialEq)]
    pub enum SolverType {
        /// Second-order Solver
        Iffbdd,
        /// First-order Solver
        Dcd,
    }

    /// Available types of constraints
    #[derive(Debug, Clone, PartialEq)]
    pub enum SenseType {
        Less,
        Greater,
        Equal,
        Interval,
    }

    /// Object to that holds info to initialize a problem
    #[derive(Debug)]
    pub struct ProblemConfig {
        /// Which solver to use
        pub solver: SolverType,
        /// Which objective to minimize
        pub objective: Objective,
        /// dimension of `x`
        pub k: usize,
        /// Contains the linear constraints
        pub constraints: Vec<ConstraintSense>,
    }

    /// method that uses [Config](struct@ProblemConfig) to initialize solver
    pub fn init_solver_from_config(config: ProblemConfig) -> Box<dyn Solver> {
        let k = config.k;
        match config.solver {
            SolverType::Iffbdd => {
                let mut s = SecondOrderSolver::new(k);
                config
                    .constraints
                    .into_iter()
                    .for_each(|c| s.add_constraint(c));
                return Box::new(s);
            }
            SolverType::Dcd => {
                let mut s = FirstOrderSolver::new(k);
                config
                    .constraints
                    .into_iter()
                    .for_each(|c| s.add_constraint(c));
                return Box::new(s);
            }
        };
    }

    /// method that parses file and returns the prepared solver
    pub fn init_from_file(
        path: impl AsRef<Path>,
    ) -> Result<(Objective, Box<dyn Solver>), ParseError> {
        let parsed_config = parse_problem_file(path)?;
        let obj = parsed_config.objective;
        Ok((obj, init_solver_from_config(parsed_config)))
    }
}

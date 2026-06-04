# Self-contained Bayesian optimization for structure-parameter search.
#
# Reference: Chen et al., "Structure-control Co-design of Quadruped
# Robots Based on Pre-training-Fine-tuning Framework", ROBOT, 2025, Eq (7-8).
#
# Uses a Gaussian Process with Matern 2.5 kernel and Expected Improvement
# acquisition to find optimal leg-length scaling factors.

import numpy as np
from scipy.linalg import cho_solve, cho_factor
from scipy.stats import norm
from typing import List, Tuple, Optional


class MaternKernel:
    """Matern 2.5 kernel: k(x,y) = (1 + sqrt(5)*d + 5*d^2/3) * exp(-sqrt(5)*d)
    where d = ||x - y||_2 / length_scale.
    """

    def __init__(self, length_scale: float = 1.0, output_scale: float = 1.0):
        self.length_scale = length_scale
        self.output_scale = output_scale

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between X1 and X2."""
        # X1: (n1, d), X2: (n2, d) -> return (n1, n2)
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        dist = np.sqrt(np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1))
        d = dist / self.length_scale
        sqrt5 = np.sqrt(5.0)
        K = self.output_scale * (1.0 + sqrt5 * d + 5.0 * d**2 / 3.0) * np.exp(-sqrt5 * d)
        return K


class GaussianProcess:
    """Gaussian Process regression.

    X: training inputs  (n, d)
    y: training outputs (n,)
    kernel: MaternKernel instance
    noise: observation noise variance
    """

    def __init__(self, kernel: MaternKernel = None, noise: float = 1e-6):
        self.kernel = kernel or MaternKernel()
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.L = None   # Cholesky factor of (K + noise*I)
        self.alpha = None  # K^{-1} y

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the GP to training data."""
        self.X_train = np.atleast_2d(X).copy()
        self.y_train = np.atleast_1d(y).copy()
        K = self.kernel(self.X_train, self.X_train)
        K += self.noise * np.eye(len(self.X_train))
        self.L = cho_factor(K, lower=True)
        self.alpha = cho_solve(self.L, self.y_train)

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and standard deviation at test points.

        Returns
        -------
        mu : ndarray (n_test,)
            Predictive mean.
        sigma : ndarray (n_test,)
            Predictive standard deviation.
        """
        if self.X_train is None:
            raise RuntimeError("GP must be fit before prediction")
        X_test = np.atleast_2d(X_test)
        K_s = self.kernel(self.X_train, X_test)       # (n_train, n_test)
        K_ss = self.kernel(X_test, X_test)             # (n_test, n_test)
        mu = K_s.T @ self.alpha
        v = cho_solve(self.L, K_s)                     # (n_train, n_test)
        cov = K_ss - K_s.T @ v
        sigma = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        return mu, sigma

    @property
    def y_max(self):
        return np.max(self.y_train) if self.y_train is not None else None


def expected_improvement(X: np.ndarray, gp: GaussianProcess,
                         y_best: float, xi: float = 0.01) -> np.ndarray:
    """Expected Improvement acquisition function.

    EI(x) = (mu - y_best - xi) * Phi(Z) + sigma * phi(Z)
    where Z = (mu - y_best - xi) / sigma

    Parameters
    ----------
    X : ndarray (n_candidates, d)
        Candidate points.
    gp : GaussianProcess
        Fitted GP.
    y_best : float
        Best observed value so far.
    xi : float
        Exploration parameter.

    Returns
    -------
    ei : ndarray (n_candidates,)
        Expected improvement at each candidate.
    """
    mu, sigma = gp.predict(X)
    # Avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    Z = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return ei


def latin_hypercube_sample(n_samples: int, bounds: List[Tuple[float, float]],
                           seed: int = None) -> np.ndarray:
    """Generate Latin Hypercube Samples.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    bounds : list of (lower, upper)
        Bounds for each dimension.
    seed : int or None
        Random seed.

    Returns
    -------
    samples : ndarray (n_samples, n_dims)
    """
    rng = np.random.RandomState(seed)
    n_dims = len(bounds)
    samples = np.zeros((n_samples, n_dims))
    for i in range(n_dims):
        lower, upper = bounds[i]
        # Generate evenly spaced intervals
        intervals = np.linspace(0, 1, n_samples + 1)
        # Randomly permute intervals within each dimension
        rand_offsets = rng.uniform(0, 1.0 / n_samples, n_samples)
        samples[:, i] = lower + (upper - lower) * (
            intervals[:-1] + rand_offsets
        )
    # Shuffle to mix dimensions
    for i in range(n_dims):
        rng.shuffle(samples[:, i])
    return samples


class BayesianOptimizer:
    """Bayesian optimization for structure parameter search.

    Solves Eq (8): xi* = arg max_{xi in Xi} f(xi | kappa, pi_xi)

    Usage
    -----
    bo = BayesianOptimizer(bounds=[(0.6, 1.4)] * 4, n_init=5, n_iter=20)
    for _ in range(bo.n_total):
        xi = bo.suggest()
        fitness = evaluate_fitness(xi)  # User-provided (Eq 7)
        bo.update(xi, fitness)
    xi_optimal = bo.best_params
    """

    def __init__(self,
                 bounds: List[Tuple[float, float]],
                 n_init: int = 5,
                 n_iter: int = 20,
                 length_scale: float = 0.3,
                 noise: float = 1e-3,
                 seed: int = None):
        """
        Parameters
        ----------
        bounds : list of (lower, upper)
            Parameter bounds for each dimension.
        n_init : int
            Number of initial random samples.
        n_iter : int
            Number of BO iterations.
        length_scale : float
            GP kernel length scale.
        noise : float
            GP observation noise.
        seed : int or None
            Random seed.
        """
        self.bounds = list(bounds)
        self.n_dims = len(bounds)
        self.n_init = n_init
        self.n_iter = n_iter
        self.n_total = n_init + n_iter

        self.kernel = MaternKernel(length_scale=length_scale)
        self.gp = GaussianProcess(kernel=self.kernel, noise=noise)
        self.rng = np.random.RandomState(seed)

        self.X_observed: List[np.ndarray] = []
        self.y_observed: List[float] = []

        # Track improvement
        self._iteration = 0

    def suggest(self) -> np.ndarray:
        """Suggest the next parameter vector to evaluate.

        Returns
        -------
        xi : ndarray (n_dims,)
            Next candidate parameter vector.
        """
        if len(self.X_observed) < self.n_init:
            # Initialization phase: Latin Hypercube Sampling
            if not hasattr(self, '_init_candidates'):
                n_remaining = self.n_init - len(self.X_observed)
                # Generate candidates; if already sampled some, adjust
                self._init_candidates = latin_hypercube_sample(
                    self.n_init, self.bounds, seed=self.rng.randint(0, 2**31)
                )
            xi = self._init_candidates[len(self.X_observed)]
        else:
            # BO phase: maximize expected improvement
            n_candidates = 2000
            X_cand = latin_hypercube_sample(
                n_candidates, self.bounds,
                seed=self.rng.randint(0, 2**31)
            )
            y_best = max(self.y_observed)
            ei = expected_improvement(X_cand, self.gp, y_best)
            best_idx = np.argmax(ei)
            xi = X_cand[best_idx]

        return xi.copy()

    def update(self, xi: np.ndarray, fitness: float):
        """Update the GP with a new observation.

        Parameters
        ----------
        xi : ndarray (n_dims,)
            Parameter vector that was evaluated.
        fitness : float
            Observed fitness value (Eq 7).
        """
        xi_arr = np.atleast_1d(np.asarray(xi, dtype=float))
        self.X_observed.append(xi_arr)
        self.y_observed.append(float(fitness))

        if len(self.X_observed) >= self.n_init:
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            # Normalize y for numerical stability
            self._y_mean = y.mean()
            self._y_std = y.std() or 1.0
            y_norm = (y - self._y_mean) / self._y_std
            self.gp.fit(X, y_norm)

        self._iteration += 1

    @property
    def best_params(self) -> Optional[np.ndarray]:
        """Return the best parameter vector found so far."""
        if not self.X_observed:
            return None
        best_idx = np.argmax(self.y_observed)
        return self.X_observed[best_idx].copy()

    @property
    def best_fitness(self) -> Optional[float]:
        """Return the best fitness value observed so far."""
        if not self.y_observed:
            return None
        return max(self.y_observed)

    @property
    def history(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return full optimization history: (X, y)."""
        return np.array(self.X_observed), np.array(self.y_observed)

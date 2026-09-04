"""Is the filter's confidence honest?

Every test in ``test_filters.py`` asks whether the estimate is *accurate*. This
module asks a different and less obvious question: whether the estimate's stated
*uncertainty* is correct. Those come apart, and the way they come apart is
dangerous.

A Kalman filter reports a covariance alongside its estimate — its own claim
about how wrong it expects to be. Nothing forces that claim to be true. A filter
whose process noise is too small tracks beautifully in the quiet and reports a
tiny covariance, which makes its gain small, which makes it slow to believe a
measurement that disagrees with it. When the target finally manoeuvres, the
filter rejects the evidence because it is too sure of itself. It does not fail
loudly; it fails by being confidently late, and the miss distance is the first
symptom anyone sees.

**NEES** — normalised estimation error squared — is the standard check. Take the
true state, subtract the estimate, and measure the error in units of the filter's
own claimed uncertainty:

    epsilon = (x - x_hat)^T P^-1 (x - x_hat)

If ``P`` is honest, ``epsilon`` is chi-squared distributed with as many degrees
of freedom as there are states, so it should average to the number of states.
Much larger means the filter is **overconfident**: its real errors are bigger
than it admits. Much smaller means it is **conservative**: it is exaggerating its
own uncertainty, which is wasteful but safe.

The asymmetry matters, and this project treats it asymmetrically. Conservative
costs performance. Overconfident costs the intercept.

**One run proves nothing.** A single trajectory gives one sample of a random
variable, and chi-squared has a long tail. Consistency is a property of the
*ensemble*, so it is measured across many independent runs at the same instant,
which is what :func:`chi_squared_interval` sizes its bounds for.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from interceptor.core.state import Vector
from interceptor.sensing.filters import TargetEstimate

__all__ = [
    "chi_squared_interval",
    "ensemble_average",
    "normalised_error_squared",
]

#: Standard normal quantiles for the two-sided confidences worth offering.
#: Tabulated rather than computed: the inverse normal CDF is not in the standard
#: library, and pulling in scipy to obtain three constants would add a large
#: dependency to a package whose only requirement is numpy.
_Z: Final[dict[float, float]] = {
    0.90: 1.6448536269514722,
    0.95: 1.9599639845400545,
    0.99: 2.5758293035489004,
}


def normalised_error_squared(
    estimate: TargetEstimate,
    position: Vector,
    velocity: Vector,
    acceleration: Vector | None = None,
) -> float:
    """NEES of one estimate against the truth it was trying to find.

    Args:
        estimate: The filter's output. Must carry a covariance — a fixed-gain
            filter has no notion of its own uncertainty, so the question this
            function asks is not defined for one.
        position: True target position, world frame.
        velocity: True target velocity, world frame.
        acceleration: True target acceleration. Supply it to test all nine
            states; omit it to test position and velocity only, which is the
            more common choice because the acceleration state of a
            constant-acceleration model is a modelling fiction rather than
            something the target possesses.

    Returns:
        The statistic, to be compared against :func:`chi_squared_interval` for 6
        or 9 degrees of freedom respectively. Its expected value under a correct
        covariance is exactly that number of degrees of freedom.

    Raises:
        ValueError: If the estimate has no covariance.
    """
    if estimate.covariance is None:
        msg = (
            "NEES needs a covariance, and this estimate has none. Fixed-gain "
            "filters such as AlphaBeta do not maintain one, so consistency is "
            "not a question that can be asked of them."
        )
        raise ValueError(msg)

    if acceleration is None:
        error = np.concatenate((position - estimate.position, velocity - estimate.velocity))
        covariance = estimate.covariance[0:6, 0:6]
    else:
        error = np.concatenate(
            (
                position - estimate.position,
                velocity - estimate.velocity,
                acceleration - estimate.acceleration,
            )
        )
        covariance = estimate.covariance[0:9, 0:9]

    # solve() rather than inv(): the same answer, but without forming an
    # explicit inverse of a matrix whose condition number spans the ratio
    # between metres of position doubt and m/s^2 of acceleration doubt.
    return float(error @ np.linalg.solve(covariance, error))


def chi_squared_interval(dof: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided acceptance interval for a chi-squared statistic.

    Args:
        dof: Degrees of freedom. For an ensemble of ``N`` independent runs each
            contributing ``n`` states, this is ``N * n`` — averaging ``N``
            samples narrows the interval, which is the entire reason a
            consistency check needs more than one run to say anything.
        confidence: Two-sided confidence. One of 0.90, 0.95 or 0.99.

    Returns:
        ``(lower, upper)`` bounds on the *sum*. Divide both by ``N`` to bound
        the average, which is what :func:`ensemble_average` returns.

    Computed by the Wilson-Hilferty transformation, which says that the cube
    root of a chi-squared variable divided by its degrees of freedom is very
    nearly normal. Checked against exact quantiles: the relative error is under
    3e-4 for 54 or more degrees of freedom and under 2e-5 by 225, which is far
    finer than the difference any real filter's tuning would make. It degrades
    in the lower tail for very small dof — 2.5% at dof = 6 — so a single-sample
    check is not what this is for, and that is also the case the docstring above
    argues against on statistical grounds anyway.
    """
    if dof < 1:
        msg = f"dof must be at least 1, got {dof}"
        raise ValueError(msg)
    if confidence not in _Z:
        msg = f"confidence must be one of {sorted(_Z)}, got {confidence}"
        raise ValueError(msg)

    z = _Z[confidence]
    scale = 2.0 / (9.0 * dof)
    lower = dof * (1.0 - scale - z * np.sqrt(scale)) ** 3
    upper = dof * (1.0 - scale + z * np.sqrt(scale)) ** 3
    return float(lower), float(upper)


def ensemble_average(
    samples: Vector, states: int = 6, confidence: float = 0.95
) -> tuple[float, tuple[float, float]]:
    """Mean NEES over independent runs, with the interval it should sit in.

    Args:
        samples: One NEES value per independent run, all taken at the same
            instant and each from its own random seed. Runs must be independent:
            sampling one trajectory repeatedly through time gives correlated
            values and a bound that is far too tight.
        states: How many states each sample covers — 6 for position and
            velocity, 9 with acceleration. Must match what was passed to
            :func:`normalised_error_squared`.
        confidence: Two-sided confidence for the returned bounds.

    Returns:
        ``(mean, (lower, upper))``, with the bounds already divided by the
        number of runs so that they apply directly to the mean. The mean should
        land near ``states``; the interval says how near is near enough.
    """
    runs = int(samples.size)
    if runs < 1:
        msg = "need at least one sample"
        raise ValueError(msg)
    lower, upper = chi_squared_interval(runs * states, confidence)
    return float(np.mean(samples)), (lower / runs, upper / runs)

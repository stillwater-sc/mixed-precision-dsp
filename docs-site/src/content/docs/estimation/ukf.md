---
title: Unscented Kalman Filter
description: Sigma-point state estimation without Jacobians in mixed-precision DSP
---

The Unscented Kalman Filter (UKF) avoids the Jacobian linearization of the
EKF by propagating a carefully chosen set of sample points -- called sigma
points -- through the nonlinear system model. This captures the mean and
covariance of the transformed distribution to at least second-order
accuracy, compared to the EKF's first-order linearization.

## The unscented transform

Given a state vector $x \in \mathbb{R}^n$ with mean $\hat{x}$ and
covariance $P$, the unscented transform generates $2n+1$ sigma points:

$$
\mathcal{X}_0 = \hat{x}
$$

$$
\mathcal{X}_i = \hat{x} + \left(\sqrt{(n + \lambda) P}\right)_i, \quad i = 1, \dots, n
$$

$$
\mathcal{X}_{i+n} = \hat{x} - \left(\sqrt{(n + \lambda) P}\right)_i, \quad i = 1, \dots, n
$$

where $(\sqrt{M})_i$ denotes the $i$-th column of the matrix square root,
and $\lambda = \alpha^2(n + \kappa) - n$ is the composite scaling parameter.

### Scaling parameters

The Julier/Merwe parameterization uses three tuning constants:

- **$\alpha$** -- Controls the spread of sigma points around the mean.
  Typically $10^{-4} \leq \alpha \leq 1$. Smaller values keep sigma points
  closer to the mean.
- **$\beta$** -- Incorporates prior knowledge of the distribution.
  $\beta = 2$ is optimal for Gaussian distributions.
- **$\kappa$** -- Secondary scaling parameter. Often set to $0$ or $3 - n$.

### Weight computation

Each sigma point carries a weight for computing the mean and covariance of
the transformed distribution:

$$
W_0^{(m)} = \frac{\lambda}{n + \lambda}
$$

$$
W_0^{(c)} = \frac{\lambda}{n + \lambda} + (1 - \alpha^2 + \beta)
$$

$$
W_i^{(m)} = W_i^{(c)} = \frac{1}{2(n + \lambda)}, \quad i = 1, \dots, 2n
$$

The superscripts $(m)$ and $(c)$ distinguish weights used for mean
recovery from those used for covariance recovery.

## Predict step

Each sigma point is propagated through the nonlinear state transition:

$$
\mathcal{X}_{k|k-1}^{(i)} = f\bigl(\mathcal{X}_{k-1}^{(i)}, u_{k-1}\bigr)
$$

The predicted mean and covariance are recovered from the transformed sigma
points:

$$
\hat{x}_k^- = \sum_{i=0}^{2n} W_i^{(m)} \mathcal{X}_{k|k-1}^{(i)}
$$

$$
P_k^- = \sum_{i=0}^{2n} W_i^{(c)} \bigl(\mathcal{X}_{k|k-1}^{(i)} - \hat{x}_k^-\bigr)\bigl(\mathcal{X}_{k|k-1}^{(i)} - \hat{x}_k^-\bigr)^T + Q
$$

## Update step

New sigma points are generated from the predicted state, then passed
through the observation model $h(\cdot)$. The measurement mean, innovation
covariance, and cross-covariance are computed, and the state is updated
with the standard Kalman gain formula.

## Matrix square root: LDL^T decomposition

Computing the sigma points requires a matrix square root of $P$. The
library uses $LDL^T$ decomposition rather than Cholesky factorization:

$$
P = L D L^T
$$

where $L$ is unit lower triangular and $D$ is diagonal. This factorization
is more numerically stable because it avoids taking square roots of
diagonal elements, which can fail when $P$ has small eigenvalues -- a
common occurrence with limited-precision arithmetic.

## Library API

```cpp
#include <sw/dsp/estimation/ukf.hpp>

using namespace sw::dsp;
using T = double;
using Vec = mtl::vec::dense_vector<T>;

UnscentedKalmanFilter<T> ukf(4, 2);  // 4 states, 2 measurements

// Scaling parameters (defaults: alpha=1e-3, beta=2, kappa=0)
ukf.alpha() = 1e-3;
ukf.beta()  = 2.0;
ukf.kappa() = 0.0;

// Nonlinear models (same interface as EKF)
ukf.f() = [](const Vec& x, const Vec& u) -> Vec { /* ... */ };
ukf.h() = [](const Vec& x) -> Vec { /* ... */ };

// No Jacobian callbacks needed

ukf.Q() = /* process noise covariance */;
ukf.R() = /* measurement noise covariance */;
ukf.P() = /* initial covariance */;
ukf.state() = /* initial state estimate */;

ukf.predict(u);
ukf.update(z);
```

## Advantages over the EKF

| Property                  | EKF                         | UKF                          |
|---------------------------|-----------------------------|------------------------------|
| Linearization accuracy    | First-order (Jacobian)      | Second-order (sigma points)  |
| Jacobian required         | Yes                         | No                           |
| Highly nonlinear systems  | Can diverge                 | More robust                  |
| Computational cost        | $O(n^3)$ Jacobian + update  | $O(n^3)$ sigma point spread  |
| Implementation complexity | Requires analytic Jacobians | Only needs $f$ and $h$       |

For mildly nonlinear systems the EKF and UKF give similar results. The UKF
shows clear advantages when the observation model involves strong
nonlinearities such as angle wrapping, coordinate transforms, or
saturation.

## Mixed-precision considerations

The sigma point spread $\sqrt{(n+\lambda)P}$ amplifies any numerical
errors in the covariance matrix. In low-precision arithmetic, the
decomposition of $P$ can produce sigma points that poorly represent the
true distribution. Using a wider type for the covariance and sigma point
computations while keeping measurements in a narrower type is recommended:

```cpp
using State  = double;                        // covariance & sigma points
using Sample = sw::universal::posit<16, 2>;   // measurements

UnscentedKalmanFilter<State> ukf(4, 2);
// Measurements promoted to State precision internally
```

The $LDL^T$ decomposition used by the library further mitigates precision
loss compared to implementations that rely on Cholesky, making the UKF
well-suited for exploration with alternative number systems.

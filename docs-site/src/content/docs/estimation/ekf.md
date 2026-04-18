---
title: Extended Kalman Filter
description: Nonlinear state estimation using Jacobian linearization in mixed-precision DSP
---

The Extended Kalman Filter (EKF) generalizes the Kalman filter to nonlinear
systems by linearizing the dynamics and observation models around the
current state estimate. It is the most widely used nonlinear state estimator
in practice, from GPS receivers to robot localization.

## Nonlinear system model

The system is described by nonlinear functions rather than matrices:

$$
x_{k+1} = f(x_k, u_k) + w_k
$$

$$
z_k = h(x_k) + v_k
$$

where $f(\cdot)$ is the state transition function, $h(\cdot)$ is the
observation function, and the noise terms remain Gaussian:
$w_k \sim \mathcal{N}(0, Q)$, $v_k \sim \mathcal{N}(0, R)$.

## Predict step

The state is propagated through the nonlinear model directly, but the
covariance is propagated through the Jacobian linearization:

$$
\hat{x}_k^- = f(\hat{x}_{k-1}, u_{k-1})
$$

$$
F_k = \left. \frac{\partial f}{\partial x} \right|_{\hat{x}_{k-1}, u_{k-1}}
$$

$$
P_k^- = F_k P_{k-1} F_k^T + Q
$$

The Jacobian $F_k$ is re-evaluated at every time step since the
linearization point changes with the state estimate.

## Update step

The observation Jacobian is computed at the predicted state:

$$
H_k = \left. \frac{\partial h}{\partial x} \right|_{\hat{x}_k^-}
$$

The standard Kalman update equations then apply with the linearized $H_k$:

$$
K_k = P_k^- H_k^T (H_k P_k^- H_k^T + R)^{-1}
$$

$$
\hat{x}_k = \hat{x}_k^- + K_k \bigl(z_k - h(\hat{x}_k^-)\bigr)
$$

$$
P_k = (I - K_k H_k) P_k^-
$$

Note that the innovation uses $h(\hat{x}_k^-)$, not $H_k \hat{x}_k^-$.
The nonlinear observation function is applied directly to compute the
predicted measurement.

## Library API

The `ExtendedKalmanFilter<T>` class uses `std::function` callbacks for the
nonlinear models and their Jacobians:

```cpp
#include <sw/dsp/estimation/ekf.hpp>

using namespace sw::dsp;
using T = double;
using Vec = mtl::vec::dense_vector<T>;
using Mat = mtl::mat::dense2D<T>;

ExtendedKalmanFilter<T> ekf(3, 2);  // 3 states, 2 measurements

// State transition function
ekf.f() = [](const Vec& x, const Vec& u) -> Vec {
    Vec x_next(3);
    // nonlinear dynamics here
    return x_next;
};

// State transition Jacobian
ekf.F_jacobian() = [](const Vec& x, const Vec& u) -> Mat {
    Mat F(3, 3);
    // partial derivatives of f w.r.t. x
    return F;
};

// Observation function
ekf.h() = [](const Vec& x) -> Vec {
    Vec z(2);
    // nonlinear observation here
    return z;
};

// Observation Jacobian
ekf.H_jacobian() = [](const Vec& x) -> Mat {
    Mat H(2, 3);
    // partial derivatives of h w.r.t. x
    return H;
};
```

Noise covariances and initial conditions are set the same way as the
linear Kalman filter:

```cpp
ekf.Q() = /* process noise covariance */;
ekf.R() = /* measurement noise covariance */;
ekf.P() = /* initial covariance */;
ekf.state() = /* initial state estimate */;

ekf.predict(u);
ekf.update(z);
```

## Example: bearing-range tracking

Consider tracking a target from a stationary sensor that measures bearing
$\theta$ and range $r$. The state is the Cartesian position and velocity:
$x = [p_x, \; p_y, \; v_x, \; v_y]^T$.

```cpp
#include <sw/dsp/estimation/ekf.hpp>
#include <cmath>

using namespace sw::dsp;
using T = double;
using Vec = mtl::vec::dense_vector<T>;
using Mat = mtl::mat::dense2D<T>;

constexpr T dt = 0.1;

ExtendedKalmanFilter<T> ekf(4, 2);

// Constant-velocity state transition (linear in this case)
ekf.f() = [](const Vec& x, const Vec& u) -> Vec {
    Vec xn(4);
    xn[0] = x[0] + x[2] * dt;  // px + vx*dt
    xn[1] = x[1] + x[3] * dt;  // py + vy*dt
    xn[2] = x[2];               // vx
    xn[3] = x[3];               // vy
    return xn;
};

ekf.F_jacobian() = [](const Vec& x, const Vec& u) -> Mat {
    Mat F(4, 4, T(0));
    F[0][0] = 1; F[0][2] = dt;
    F[1][1] = 1; F[1][3] = dt;
    F[2][2] = 1;
    F[3][3] = 1;
    return F;
};

// Nonlinear observation: bearing and range
ekf.h() = [](const Vec& x) -> Vec {
    Vec z(2);
    z[0] = std::atan2(x[1], x[0]);                     // bearing
    z[1] = std::sqrt(x[0]*x[0] + x[1]*x[1]);           // range
    return z;
};

ekf.H_jacobian() = [](const Vec& x) -> Mat {
    T px = x[0], py = x[1];
    T r2 = px*px + py*py;
    T r  = std::sqrt(r2);
    Mat H(2, 4, T(0));
    H[0][0] = -py / r2;   H[0][1] = px / r2;   // d(bearing)/d(px,py)
    H[1][0] =  px / r;    H[1][1] = py / r;     // d(range)/d(px,py)
    return H;
};
```

## Numerical issues with Jacobian computation

The EKF's accuracy depends critically on the quality of the Jacobian
matrices. Two key concerns arise in mixed-precision settings:

1. **Analytic vs. numerical Jacobians.** Analytic derivatives are preferred
   for accuracy. When using finite-difference approximations, the
   perturbation step $\delta$ must be chosen carefully -- too small and
   floating-point cancellation dominates; too large and truncation error
   grows.

2. **Precision of intermediate computations.** Jacobian entries involve
   divisions and transcendental functions that amplify rounding errors.
   Using a wider arithmetic type for Jacobian evaluation preserves EKF
   convergence:

```cpp
using State = double;                        // Jacobian & covariance
using Sample = sw::universal::posit<16, 2>;  // measurements

ExtendedKalmanFilter<State> ekf(4, 2);
// Measurements are promoted to State precision before the update
```

The library provides a numerical Jacobian utility for prototyping:

```cpp
// Compute Jacobian of f at point x by central differences
auto F_num = estimation::numerical_jacobian(f, x, T(1e-8));
```

This should be replaced with analytic Jacobians in production code where
the perturbation step cannot be tuned for all operating regimes.

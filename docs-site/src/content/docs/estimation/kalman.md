---
title: Kalman Filter
description: Linear state estimation with the Kalman filter in mixed-precision DSP
---

The Kalman filter is the optimal linear state estimator for systems with
Gaussian noise. It recursively estimates the state of a dynamic system from
a series of noisy measurements, balancing prediction uncertainty against
measurement uncertainty via the Kalman gain.

## System model

A linear discrete-time system is described by:

$$
x_{k+1} = F x_k + B u_k + w_k
$$

$$
z_k = H x_k + v_k
$$

where $x_k$ is the state vector, $u_k$ is the control input, $z_k$ is the
measurement, $w_k \sim \mathcal{N}(0, Q)$ is process noise, and
$v_k \sim \mathcal{N}(0, R)$ is measurement noise.

## Predict step

The filter propagates the state estimate and covariance forward in time:

$$
\hat{x}_k^- = F \hat{x}_{k-1} + B u_{k-1}
$$

$$
P_k^- = F P_{k-1} F^T + Q
$$

The predicted covariance $P^-$ grows by the process noise $Q$ at each step,
reflecting increasing uncertainty between measurements.

## Update step

When a measurement $z_k$ arrives, the filter computes the Kalman gain and
corrects the prediction:

$$
K_k = P_k^- H^T (H P_k^- H^T + R)^{-1}
$$

$$
\hat{x}_k = \hat{x}_k^- + K_k (z_k - H \hat{x}_k^-)
$$

$$
P_k = (I - K_k H) P_k^-
$$

The term $z_k - H \hat{x}_k^-$ is the innovation (measurement residual).
When $R$ is large relative to $P^-$, the gain shrinks and the filter trusts
the prediction more than the measurement.

## Library API

The `KalmanFilter<T>` class template provides a complete implementation:

```cpp
#include <sw/dsp/estimation/kalman.hpp>

using namespace sw::dsp;
using T = double;

// 4-state, 2-measurement system
KalmanFilter<T> kf(4, 2);

// Access system matrices
kf.F() = /* state transition matrix */;
kf.H() = /* observation matrix */;
kf.Q() = /* process noise covariance */;
kf.R() = /* measurement noise covariance */;
kf.B() = /* control input matrix */;
kf.P() = /* initial state covariance */;
kf.state() = /* initial state estimate */;
```

The `predict()` and `update()` methods execute the two-step cycle:

```cpp
mtl::vec::dense_vector<T> u(2);  // control input
mtl::vec::dense_vector<T> z(2);  // measurement

kf.predict(u);   // propagate state and covariance
kf.update(z);    // correct with measurement
```

## Example: constant-velocity tracking

A common application is tracking an object in one dimension with position
and velocity as state variables: $x = [p, \; v]^T$.

```cpp
#include <sw/dsp/estimation/kalman.hpp>

using namespace sw::dsp;
using T = double;

constexpr T dt = 0.1;  // 10 Hz sample rate

KalmanFilter<T> tracker(2, 1);  // 2 states, 1 measurement

// State transition: constant velocity model
// p_{k+1} = p_k + v_k * dt
// v_{k+1} = v_k
mtl::mat::dense2D<T> F(2, 2);
F = { {1.0, dt},
      {0.0, 1.0} };
tracker.F() = F;

// Observation: we measure position only
mtl::mat::dense2D<T> H(1, 2);
H = { {1.0, 0.0} };
tracker.H() = H;

// Process noise: acceleration disturbance
T q = 0.1;
mtl::mat::dense2D<T> Q(2, 2);
Q = { {dt*dt*dt*dt/4*q, dt*dt*dt/2*q},
      {dt*dt*dt/2*q,     dt*dt*q} };
tracker.Q() = Q;

// Measurement noise
mtl::mat::dense2D<T> R(1, 1);
R = { {1.0} };  // 1 m^2 variance
tracker.R() = R;

// Initial covariance (high uncertainty)
mtl::mat::dense2D<T> P(2, 2);
P = { {10.0, 0.0},
      {0.0,  10.0} };
tracker.P() = P;

// Run filter loop
for (std::size_t k = 0; k < measurements.size(); ++k) {
    tracker.predict();
    tracker.update(measurements[k]);

    auto est = tracker.state();
    // est[0] = estimated position
    // est[1] = estimated velocity
}
```

## Mixed-precision considerations

The covariance update $P = (I - KH)P^-$ involves subtraction of
nearly-equal matrices, which can cause catastrophic cancellation in low
precision. Using a wider `StateScalar` for the covariance matrices while
keeping measurements in a narrower `SampleScalar` preserves filter
stability:

```cpp
using Sample = sw::universal::posit<16, 2>;  // measurements
using State  = double;                        // covariance math

KalmanFilter<State> kf(4, 2);
// Measurements are converted to State precision internally
```

The Joseph form of the covariance update, $P = (I-KH)P^-(I-KH)^T + KRK^T$,
is numerically more stable and is used by the library when the state
dimension exceeds the measurement dimension.

## Performance notes

| State dim $n$ | Meas dim $m$ | Dominant cost           |
|---------------|-------------|-------------------------|
| Small ($< 10$)| Small       | $O(n^2 m)$ per update   |
| Large         | Small       | Covariance propagation  |
| Large         | Large       | Innovation inverse $O(m^3)$ |

For systems with many states but few measurements, the covariance predict
step $FPF^T + Q$ dominates. The library exploits symmetry in $P$ to reduce
computation by roughly half.

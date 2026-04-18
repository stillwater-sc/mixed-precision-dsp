---
title: Adaptive Filters
description: LMS, NLMS, and RLS adaptive filtering algorithms in mixed-precision DSP
---

Adaptive filters adjust their coefficients in real time to minimize an
error signal, without requiring prior knowledge of the signal statistics.
They are the foundation of echo cancellation, channel equalization, noise
cancellation, and system identification.

## LMS: Least Mean Squares

The LMS algorithm updates a length-$p$ weight vector $w$ to minimize the
mean squared error between a desired signal $d[n]$ and the filter output
$y[n] = w^T x[n]$:

$$
e[n] = d[n] - w_n^T x[n]
$$

$$
w_{n+1} = w_n + \mu \cdot e[n] \cdot x[n]
$$

where $\mu$ is the step size (learning rate) and $x[n]$ is the input
vector of the $p$ most recent samples.

### Convergence condition

The LMS algorithm converges in the mean if the step size satisfies:

$$
0 < \mu < \frac{2}{\lambda_{\max}}
$$

where $\lambda_{\max}$ is the largest eigenvalue of the input
autocorrelation matrix $R_{xx} = E[x x^T]$. In practice, a conservative
bound uses the trace: $\mu < 2 / \mathrm{tr}(R_{xx}) = 2 / (p \cdot \sigma_x^2)$.

### Step size tradeoff

- **Small $\mu$**: slow convergence, low steady-state misadjustment
- **Large $\mu$**: fast convergence, high steady-state misadjustment

The excess mean squared error in steady state is approximately
$\mu \cdot p \cdot \sigma_v^2 / 2$, where $\sigma_v^2$ is the noise
power. This tradeoff is fundamental to all gradient-descent adaptive
filters.

## NLMS: Normalized LMS

The Normalized LMS algorithm removes the dependence of convergence speed on
the input signal power by normalizing the step size:

$$
\mu_{\text{eff}}[n] = \frac{\mu}{\|x[n]\|^2 + \epsilon}
$$

$$
w_{n+1} = w_n + \mu_{\text{eff}}[n] \cdot e[n] \cdot x[n]
$$

The regularization constant $\epsilon > 0$ prevents division by zero when
the input energy is small. With normalization, $\mu \in (0, 2)$ guarantees
convergence regardless of signal level, making NLMS the preferred choice in
most practical systems.

## RLS: Recursive Least Squares

The RLS algorithm minimizes the exponentially weighted least-squares cost:

$$
J_n = \sum_{k=0}^{n} \lambda^{n-k} |e[k]|^2
$$

where $\lambda \in (0, 1]$ is the forgetting factor. The recursion
maintains an inverse correlation matrix $P_n$ and computes:

$$
K_n = \frac{P_{n-1} x_n}{\lambda + x_n^T P_{n-1} x_n}
$$

$$
e[n] = d[n] - w_n^T x_n
$$

$$
w_{n+1} = w_n + K_n \cdot e[n]
$$

$$
P_{n+1} = \frac{1}{\lambda}\bigl(P_n - K_n x_n^T P_n\bigr)
$$

### LMS vs. RLS comparison

| Property                | LMS              | RLS                    |
|-------------------------|------------------|------------------------|
| Convergence speed       | Slow             | Fast                   |
| Computational cost      | $O(p)$ per sample | $O(p^2)$ per sample   |
| Tracking ability        | Good             | Excellent              |
| Numerical sensitivity   | Low              | High (matrix inversion)|
| Memory                  | $O(p)$           | $O(p^2)$              |

RLS converges in roughly $2p$ samples regardless of the eigenvalue spread,
while LMS convergence depends on the condition number of $R_{xx}$.

## Library API

### LMS filter

```cpp
#include <sw/dsp/estimation/adaptive.hpp>

using namespace sw::dsp;
using T = double;

constexpr std::size_t order = 32;
constexpr T mu = 0.01;

LMSFilter<T> lms(order, mu);

// Process one sample at a time
for (std::size_t n = 0; n < num_samples; ++n) {
    T input   = x[n];
    T desired = d[n];

    T output = lms.filter(input);
    T error  = lms.update(desired);
    // error = d[n] - y[n], weights updated internally
}

// Access the adapted coefficients
auto weights = lms.weights();
```

### RLS filter

```cpp
#include <sw/dsp/estimation/adaptive.hpp>

using namespace sw::dsp;
using T = double;

constexpr std::size_t order = 32;
constexpr T lambda = 0.99;     // forgetting factor
constexpr T delta  = 100.0;    // P initialization: P_0 = delta * I

RLSFilter<T> rls(order, lambda, delta);

for (std::size_t n = 0; n < num_samples; ++n) {
    T output = rls.filter(x[n]);
    T error  = rls.update(d[n]);
}

auto weights = rls.weights();
```

## Applications

### Echo cancellation

An adaptive filter models the echo path from loudspeaker to microphone.
The far-end signal is the filter input, and the microphone signal is the
desired output. The error signal -- the difference between the microphone
and the predicted echo -- is the cleaned near-end speech sent to the
far-end caller.

### Channel equalization

A communication channel introduces inter-symbol interference (ISI). An
adaptive equalizer placed after the receiver uses known training symbols
to learn the inverse channel response, then switches to decision-directed
mode for steady-state operation.

### Noise cancellation

A reference microphone picks up ambient noise correlated with the noise
component in the primary signal. The adaptive filter predicts the noise
in the primary channel and subtracts it, leaving the desired signal.

## Mixed-precision considerations

The RLS matrix update $P_{n+1} = (P_n - K_n x_n^T P_n)/\lambda$ involves
subtraction of nearly-equal matrices, much like the Kalman covariance
update. In low-precision arithmetic, $P$ can lose positive definiteness
and the filter diverges. Using a wider type for the internal state
mitigates this:

```cpp
using Sample = sw::universal::posit<16, 2>;  // I/O samples
using State  = double;                        // P matrix, gain vector

RLSFilter<State> rls(32, 0.99, 100.0);
// Input samples promoted to State precision internally
```

LMS is far less sensitive to precision because its update is a simple
vector addition with no matrix inversion. For resource-constrained systems,
LMS or NLMS with a narrow arithmetic type is often the practical choice.

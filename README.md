# Gaussian Random Field

## Description

This is a d-dimensional stationary Gaussian random field (GRF) generator, using circulant embedding method [1]. By defining the covariance function, we can generate any kind of stationary GRFs. Particularly, by choosing a specific covariance function and add a trend, we can generate fractional Brownian motion (fBm) fields [2].

## Usage

We use fBm field generation as an example. 

1. Define the covariance function as a class.
```python
class PrefbmCov:
```

2. Define parameters.
```python
hurst = 0.95
alpha = hurst * 2
c = 1
num_pt = 100
minpadding = 0
```

3. Initialize GRF generator and generate a pre-fBm field.
```python
cov = PrefbmCov(alpha)
pts=(np.linspace(0,1,num_pt),)*2
mean = np.zeros((num_pt,num_pt))
prefbm_gen = GaussianRandomFieldCircEmbed(mean, cov, pts, minpadding=256)
prefbm_field = prefbm_gen.sample()
```

4. Add a trend to make it a fbm field
```python
c2 = alpha * (5 + 2 * alpha) / 18
sigma = np.sqrt(2 * c2)
coords = np.meshgrid(*pts, indexing="ij")
X = np.random.normal(loc=0.0, scale=sigma, size=len(coords))
trend = sum(xi_coord * Xi for xi_coord, Xi in zip(coords, X))
fbm_field = prefbm_field + trend
plt.imshow(fbm_field)
```

## Ensemble Statistics Test
See `grf_generation_and_test.ipynb`.

## Acknowledgment

This project is a Python translation of the Julia implementation from:

- [GaussianRandomFields.jl](https://github.com/PieterjanRobbe/GaussianRandomFields.jl) by PieterjanRobbe

The original project is licensed under the MIT license.  
All credit for the original algorithm and implementation goes to the original authors.

## References

[1] [Wood, A. T. A., & Chan, G. (1994). *Simulation of stationary Gaussian processes in [0, 1]*$^d$. *Journal of Computational and Graphical Statistics*, **3**(4), 409–432.](https://www.tandfonline.com/doi/abs/10.1080/10618600.1994.10474655)

[2] [Stein, M. L. (2002). *Fast and exact simulation of fractional Brownian surfaces*. *Journal of Computational and Graphical Statistics*, **11**(3), 587–599.](https://www.tandfonline.com/doi/abs/10.1198/106186002466)

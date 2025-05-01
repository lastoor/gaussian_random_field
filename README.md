# Gaussian Random Field

## Description

This is a d-dimensional stationary Gaussian random field (GRF) generator, using circulant embedding method. By defining the covariance function, we can generate any kind of stationary GRFs. Particularly, by choosing a specific covariance function and add a trend, we can generate fractional Brownian motion (fBm) fields.

## Usage

We use fBm field generation as an example. 

1. Define the covariance function as a class.
'''
class PrefbmCov:
    def __init__(self, alpha):
        self.alpha = alpha
        self.is_even = True
    def __call__(self, x):
        r = np.linalg.norm(x)
        α = self.alpha
        if r == 0:
            return 1 - α/6 - α**2/6
        if r <= 1:
            return 1 - α/6 - α**2/6 - r**α + α*(5+2*α)*r**2/18
        elif r <= 2:
            return α*(2-α)/18 * (2-r)**3 / r
        else:
            return 0.0
'''

2. Define parameters.

'''
alpha = 0.5
num_pt = 512
'''

3. Initialize GRF generator and generate a pre-fBm field.
'''
cov = PrefbmCov(alpha)
pts=(np.linspace(0,1,num_pt),)*2
mean = np.zeros((num_pt,num_pt))
prefbm_gen = GaussianRandomFieldCircEmbed(mean, cov, pts, minpadding=256)
prefbm = prefbm_gen.sample()
'''

4. Add a trend to make it a fbm field
'''
c2 = alpha * (5 + 2 * alpha) / 18
sigma = np.sqrt(2 * c2)
coords = np.meshgrid(*pts, indexing="ij")
X = np.random.normal(loc=0.0, scale=sigma, size=len(coords))
trend = sum(xi_coord * Xi for xi_coord, Xi in zip(coords, X))
fbm_field = prefbm + trend
plt.imshow(fbm_field)
'''

## Ensemble Statistics Test
See `fbm_stats.ipynb`.

## Acknowledgment

This project is a Python translation of the Julia implementation from:

- GaussianRandomFields.jl (https://github.com/PieterjanRobbe/GaussianRandomFields.jl) by PieterjanRobbe

The original project is licensed under the MIT license.  
All credit for the original algorithm and implementation goes to the original authors.

## References

[1] Wood, A. T. A., & Chan, G. (1994). [Simulation of stationary Gaussian processes in $[0, 1]^d$](https://www.tandfonline.com/doi/abs/10.1080/10618600.1994.10474655). *Journal of Computational and Graphical Statistics*, **3**(4), 409–432.

[2] Stein, M. L. (2002). [Fast and exact simulation of fractional Brownian surfaces](https://www.tandfonline.com/doi/abs/10.1198/106186002466). *Journal of Computational and Graphical Statistics*, **11**(3), 587–599.

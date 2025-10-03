import jax
import jax.numpy as jnp
from distrax import MultivariateNormalDiag

@jax.jit
def renyiDivergenceMultivariateGaussians(p1: MultivariateNormalDiag, p2: MultivariateNormalDiag):
    """
    This function computes the Rényi divergence with coefficient alpha = 2 between two multivariate gaussian 
    distributions. Specifically, we are requiring each gaussian to be diagonal
    """
    # data retrieval
    mu1 = p1.loc
    var1 = jnp.power(p1.scale_diag, 2)
    
    mu2 = p2.loc
    var2 = jnp.power(p2.scale_diag, 2)
    var2 = jnp.clip(var2, a_min=var1 * 0.5)

    # compute divergence
    denominator = 2 * var2 - var1
    log_term = 0.5 * jnp.sum(jnp.log(jnp.power(var2, 2)) - jnp.log(var1) - jnp.log(denominator))
    squared_mu_diff = jnp.power(mu1 - mu2, 2)
    mean_term = jnp.sum(squared_mu_diff / denominator)

    return log_term + mean_term

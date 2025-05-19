from .nnlasso import nnLASSO
from .nnlasso_linear_penalty import nnLASSO_linear_penalty
from .nnlasso_quadratic_penalty import nnLASSO_quadratic_penalty
from .nnlasso_exponential_penalty import nnLASSO_exponential_penalty
from .nnlasso_admm import nnLASSO_ADMM

OPTIMIZEE_DICT = {
    'nnLASSO': nnLASSO,
    'nnLASSO-linear-penalty': nnLASSO_linear_penalty,
    'nnLASSO-quadratic-penalty': nnLASSO_quadratic_penalty,
    'nnLASSO-exponential-penalty': nnLASSO_exponential_penalty,
    'nnLASSO-ADMM': nnLASSO_ADMM
}
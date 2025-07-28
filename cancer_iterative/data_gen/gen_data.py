from copula_gan import make_copula_data
from gaussian import make_gaussian_data
from ct_gan import make_ctgan_data
from tvae import make_tvae_data


for p in range (10, 109, 10):
    make_gaussian_data(p)
    make_copula_data(p)
    make_ctgan_data(p)
    make_tvae_data(p)




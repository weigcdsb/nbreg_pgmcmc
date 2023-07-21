from polyagamma import random_polyagamma
import numpy as np
from pyhmc import hmc
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import gamma, digamma
from tqdm import tqdm

def CRT_sum(x,r):
    Lsum = 0
    RND = r/(r+np.arange(0,np.max(x)))
    for ii in range(x.size):
        if x[ii] > 0:
            Lsum = Lsum + np.sum(np.random.uniform(size = x[ii]) <= RND[0:x[ii]])
    
    return(Lsum)

def nbreg_mcmc(y, X, iter_num = 1000,
               b = None, B = None, a0 = 1, h = 1,
               use_hmc = True):
    
    y = y.ravel()
    
    # 1. prior set up
    q = X.shape[1]
    if b is None:
        b = np.zeros(q)
    
    if B is None:
        B = np.eye(q)*1e2
    
    # 2. allocation
    beta_trace = np.zeros((iter_num, q))
    r_trace = np.zeros(iter_num)

    # 3. initialization: by NB regression
    poisson_res = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    lam = poisson_res.mu
    aux_ols = ((y - lam)**2 - lam)/lam
    alph_res = sm.OLS(aux_ols, lam).fit()
    alph_tmp = np.abs(alph_res.params[0])
    nb2_res = sm.GLM(y, X,family=sm.families.NegativeBinomial(alpha = alph_res.params[0])).fit()

    beta_trace[0,:] = nb2_res.params
    r_trace[0] = 1/alph_tmp
    
    
    def lg_pdf(r,pp):
        llhd = np.sum(np.log(gamma(r+y.ravel())) - np.log(gamma(r)) + r*np.log(1-pp.ravel()))
        lprior = (a0-1)*np.log(r) - h*r
        logp = llhd + lprior
        
        llhd_grad = np.sum(digamma(r+y.ravel()) - digamma(r) + np.log(1-pp.ravel()))
        lprior_grad = (a0-1)/r - h
        grad = llhd_grad + lprior_grad
        
        return logp, grad


    for gg in tqdm(range(1,iter_num)):
    
        # 1. Sample PG variables: 
        # omega ~ PG(y+r, x'\beta - log{r})
        # Kappa = omega*log{r} + (y-r)/2
        # then the transformed \hat{y} = omega^{-1}\kappa \sim N(X\beta, \omega^{-1})
        omega = random_polyagamma(y + r_trace[gg-1], X@beta_trace[gg-1,:]-np.log(r_trace[gg-1]))

        Kappa = omega*np.log(r_trace[gg-1]) + (y - r_trace[gg-1])/2
        Omega = np.diag(omega)

        # 2. update beta:
        # \beta|- \sim N(m, V)
        # V = (X'\Omega X + B^{-1})^{-1}
        # m = V(X'\Kappa + B^{-1}*b)
        V_inv = X.T @ Omega @ X + np.linalg.inv(B)
        V = np.linalg.inv(V_inv)
        m = V @ (X.T @ Kappa + np.linalg.inv(B) @ b)
        beta_trace[gg,:] = np.random.multivariate_normal(m, V)

        # 3. update r: 
        r_tmp = r_trace[gg-1]
        mu_tmp = np.exp(X @ beta_trace[gg,:])

        if use_hmc:
            # HMC
            p_tmp = mu_tmp/(mu_tmp+r_tmp)
            lg_pdf_tmp = lambda r: lg_pdf(r, p_tmp)
            samples = hmc(lg_pdf_tmp, x0=np.array([r_tmp]), n_samples=1)
            r_trace[gg] = samples


        else:
            # follow Zhou et al., 2013
    #         h = np.random.gamma(a0 + b0, 1/(g0 + r_tmp))
            Lsum = CRT_sum(y,r_tmp)
            r_trace[gg] = np.random.gamma(a0 + Lsum, 1/(h-np.sum(np.log(r_tmp/(r_tmp + mu_tmp)))))
            
    return beta_trace, r_trace     
        
    
## Note:
The codebase is not yet organised for systematic benchmarking. At this stage, I am prioritising the derivation and implementation of core approximate inference components. The current workflow uses logistic regression as a test case, however, it will be extended to higher-dimensional models (eg: Bayesian Neural networks and flow-based models) as well as other approximate inference methods. Once the foundations have been built out, I will restructure for formal benchmarking. 

## Method:

1. Data Generation:
- Domain 1: baseline
- Domain 2: shifted dataset obtained via covariate transform of X

2. Fit mean-field VI to Domain 1 to obtain posterior mean of Domain 1

3. Calculate Hessian Vector Products around mean-field VI using Lanczos to get top eigenvectors/eigenvalues and define a low-rank curvature subspace

4. Fit a whitened, Hessian-aligned low-rank Guassian VI posterior using the curvature basis

5. Distribution shift applied (Domain 1 to Domain 2)

6. Low-rank VI in Domain 2
   
   (a) Online LR-VI:
       Warm-start from the Domain 1 posterior and periodically refresh curvature 
       using local Hessian–vector products

   (b) Cold-start LR-VI:
       Initialise randomly and optimise on Domain 2 


8. Domain 2 sampled with MALA, warm-started from the online low-rank VI mean, to obtain a high-quality posterior for reference

9. Diagnositcs
 - posterior error mean vs MALA
 - principal angles between Domain 1 and Domain 2 curvature subspaces
 - principal angles between VI covariances and Domain 2 curvature
 - scale test

## Diagnostics:

**Top curvature domain1:** 
- Domain 1: 193.5265, 201.1939, 212.8611, 220.5381, 223.2889
- Domain 2: 1059.1334, 1325.9392, 1542.9922, 2461.9526, 2965.0952

**Curvature subspace drift (principal angles):** 
0.1782, 0.2861, 0.3802, 0.6641, 0.8949, 1.0148, 1.1134, 1.2136, 1.3430, 1.4998

**Posterior mean shift (Domain 1 LR-VI to Domain 2:**
- Online: 0.1306
- Cold-start: 0.5146

**Mean error vs MALA (Domain 2):**
- Online LR-VI: 0.3634
- Cold-start LR-VI: 0.5655
- LR-VI Domain 1: 0.4690
- MFVI Domain 1: 0.5025

**Covariance-Hessian subspace angles:**
0.39 to 1.52 radians

**Scale test ('i'th eigenvalue of Hessian * variance of q(w)):**
Strong under-dispersion in leading curvature directions and over-dispersion in weaker ones


## Interpretation:
The covariate shift induced a large change in posterior between domains. Domain 2 exhibits stronger curvature with leading Hessian eigenvalues being roughly an order of magnitude greater than in Domain 1. The curvature also rotates, indicating dominant directions in Domain 2 are not aligned with Domain 1. 

Online low-rank VI adapts more efficiently than using a cold start, but still diverges from the MALA posterior. The curvature subspace rotates under the shift, and the inherited low-rank basis no longer captures the dominant modes, as reflected in the covariance–Hessian angles (0.39–1.52 rad). The scale test confirms the mismatch with under-dispersion in high-curvature directions and over-dispersion in weak ones.

Increasing rank or implementing a more adaptive basis tracking scheme could support the capture of posterior geometry after distribution shift.

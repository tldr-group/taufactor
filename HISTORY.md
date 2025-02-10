# History

## 1.2.0 (2025-02-10)

---

- Add paper to README 
- Exit solve after iter_limit reached (wasn't enforced before) by @amirDahari1 in #96
- Change error for negative flux (changed stop criteria) resolves #94 by @amirDahari1 in #95
- Calc vertical flux function (returns the flux field) by @amirDahari1 in #98
- Deleted unused variables by @amirDahari1 in #103
- Use inheritance for check_vertical_flux (cleanup) by @daubners in #104
- Add AnisotropySolver (if the z-spacing is different than x- or y-spacing) by @daubners in #105
- New surface areas (face counting, gradient and marching cubes) by @daubners in #107
- Through connectivity (checks if there's phase connectivity) by @daubners in #106

## 1.1.0 (2023-07-24)

---

-   Added comments from reviewers
-   Added examples to documentation
-   Added API documentation
-   Fix test times on comparison

## 1.0.0 (2023-03-23)

---

-   Migrated to PyTorch from CuPy
-   New convergence criteria
-   New documentation style
-   CI testing
-   Includes TauFactor paper

## 0.1.4 (2022-07-11)

---

-   Add TauE solver
-   Add triple phase boundary calculations
-   Fix cuboids not converging
-   Fix convergence messaging

## 0.1.3 (2021-03-25)

---

-   Hotfix code in taufactor.py

## 0.1.2 (2021-03-25)

---

-   Added multi-phase and periodic solvers and metrics calculations

## 0.1.1 (2021-02-10)

---

-   Removed CuPy from requirements and added installation instructions to README

## 0.1.0 (2021-02-08)

---

-   First release on PyPI.

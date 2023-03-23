---
title: "TauFactor 2: A GPU accelerated python tool for microstructural analysis"
tags:
    - Python
    - Material Science
    - Tortuosity
    - Modelling
    - Effective diffusivity
    - Bruggeman coefficient
authors:
    - name: Steve Kench
      orcid: 0000-0002-7263-6724
      equal-contrib: true
      affiliation: 1 # (Multiple affiliations must be quoted)
    - name: Isaac Squires
      equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
      orcid: 0000-0003-1919-061X
      affiliation: 1
    - name: Samuel Cooper
      corresponding: true # (This is how to denote the corresponding author)
      affiliation: 1
affiliations:
    - name: Imperial College London, UK
      index: 1
date: 9 March 2023
bibliography: paper.bib
---
\SetWatermarkText{}
# Summary

TauFactor 2 is an open-source, GPU accelerated microstructural analysis tool for extracting metrics from voxel based data, including transport properties such as the touristy factor. Tortuosity factor, $\tau$, is a material parameter that defines the reduction in transport arising from the arrangement of the phases in a multiphase medium (see \autoref{example}). As shown in \autoref{eq:tort}, the effective transport co-efficient of a material, $D_{eff}$, can be calculated from the phases intrinsic transport coefficient, $D$, volume fraction, $\epsilon$, and $\tau$ [@cooper2016taufactor]. Note, this value of $\tau$ should not be squared [@tjaden2016origin].

\begin{equation}\label{eq:tort}
D_{eff} = D\dfrac{\epsilon}{\tau}
\end{equation}

Tortuosity factor has been a metric of interest in a broad range of fields for many of decades. In geophysics, $\tau$ influences groundwater flow through pourous rocks, which has significant environmental contamination impacts [@carey2016estimating]. Electrochemists use $\tau$ to solve a reduced-order system of equations describing the electrochemical behaviour of lithium-ion batteries, which influences a cells power rating [@landesfeind2018tortuosity]. The imaging and subsequent modeling of materials to determine $\tau$ is thus commonplace.



![Microstructure and flux field of a sample from the microlib library [@kench2022microlib].\label{example}](example.pdf)

# Statement of need

Materials characterisation techniques are constantly improving, allowing the collection of larger field-of-view images with higher resolutions [@withers2021x]. Alongside these developments, machine learning algorithms have enabled the generation of arbitrarily large volumes, and can further enhance image quality through super-resolution techniques [@dahari2023fusion]. The resulting high fidelity microstructural datasets can be used to extract statistically representative metrics of a materials composition and performance. However, with increasing dataset size, the computational cost to perform such analysis can lead to prohibitively long run times. This is especially problematic for transport type metrics, such as the touristy factor, as they are inherently 3D and require the use of iterative solvers. 

TauFactor 1 [@cooper2016taufactor] provided an open-source MATLAB application for calculating various microstructural metrics, including the touristy factor. However, it’s implementation as a serial CPU based solver meant that large microstructural dataset could take hours to converge. This made TauFactor 1 unsuitable for use in high-throughput tasks such as materials optimisation. TauFactor 2 provides the necessary efficiency to ensure users can analyse large datasets in reasonable times. The software is built with PyTorch [@pytorch], a commonly used and highly optimised python package for machine learning. The GPU acceleration that has enabled the drastic speed up of neural network training proves equally effective for the task of iteratively solving transport equations, where matrix multiplication and addition are the main operations required. The use of Python and PyTorch ensures broad support and easy installation, as well as the option to run the software on CPU if GPU hardware is not available. The ability to run simulations with just a few lines of code ensures accessibility for researchers from the diverse fields where this software may be of merit.

The Python implementation is similar to the original TauFactor 1, taking advantage of efficiency gains such as the precalculation of prefactors and the use of over-relaxation. The same convergence criteria are also used, where the vertical flux between each layer is averaged across the planes parallel to the stimulated boundaries. The percentage error between the minimum and maximum flux is used to indicate whether steady state has been reached. Once, this is satsfied, an extra 100 iterations are performed to check the stability of the system. A notable difference in TauFactor 2 is that flux is calculated for all voxels. This replaces an indexing system in TauFactor 1, which solved only in active voxels. We find that the speed of GPU indexing compared to matrix multiplication makes this trade-off worthwhile. As well as the standard solver for a single transport phase, a multi-phase solver is available, where the tortuosity relates $D_{eff}$ to the intrinsic diffusion coefficients and volume fractions of the phases, $p$, as follows:

$$ D_{eff} = \dfrac{\sum_p{D_p \epsilon_p}}{\tau}$$

The numerator is a weighted sum of the active phase transport coefficients according to their volume fractions, which gives a transport coefficient equivalent to solid blocks of each phase spanning a test volume (i.e. perfectly straight transport paths). Periodic boundary conditions can also be used, which negate replace no-flux boundary conditions at the non-conductive edges. Finally, Taufactor 2 also includes an electrode tortuosity factor solver (see [@nguyen2020electrode]). There are also GPU accelerated functions for calculating volume fractions, surface areas, triple phase boundaries and the tow-point correlation function. To compare the performance of TauFactor 2 to other available software, a test volume (500x500x500 = 125,000,000 voxels) was created. One of the phases in this two-phase volume fully percolates in all three directions, while the other phase does not percolate at all. The percolating phase has a volume fraction of exactly 30%. The percolating network is anisotropic in the three directions, leading to different transport metrics. Lastly, the structure is periodic at its boundaries, allowing for the exploration of the impact of periodic transport boundaries. This microstructure is available in the GitHub repo, providing a standard against which new software can also be measured. The speed of five different solvers, namely TauFactor 1 [@cooper2016taufactor], TauFactor 1.9 (an updated version of TauFactor 1 that has new solvers, such as diffusion impedance and can be called inline as well as from the GUI [@cooper2017simulated]), TauFactor 2, TauFactor 2 CPU, PoreSpy [@gostick2019porespy] and Puma [@ferguson2018puma], are shown in \autoref{speeds}. To check the accuracy of the calaculated tortuosity factors, we overconverge TauFactor 2 to give a ‘true value’ in each direction. Using default convergence criteria, all five solvers are within 0.5% of the true values other then PuMa’s explicit jump solver (5% error), which is thus excluded. For this analysis we used a NVIDIA A6000 48GB GPU and AMD Ryzen Threadripper 3970X Gen3 32 Core TRX4 CPU. TauFactor 2 is over 10 times faster than the next best solver, TauFactor 1.9, And over 100 times faster than the original TauFactor 1 solver.


![Speed comparison for the four solvers. The mean time across all 3 directions is plotted. The values of the overconverged $\tau$ in each direction are: 1.1513, 1.3905, 4.2431. \label{speeds}](tauspeeds.pdf)

# Acknowledgements

We acknowledge contributions from Amir Dahari.

# References

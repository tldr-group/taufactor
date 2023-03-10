---
title: "Taufactor 2: A GPU accelerator Laplacian solver"
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

# Summary

TauFactor 2 is an open-source, GPU accelerated diffusion solver for the calculation of tortuosity. Tortuosity, $\tau$, is a material parameter that defines the reduction in diffusion transport arising from the curvature of flow paths through a pourous medium (see Figure \autoref{tauexample}). As shown in equation \autoref{eq:tort}, the effective diffusion co-efficient of a multiphase material, $D_eff$, can thus be calculated from the diffusive phases intrinsic diffusivity, $D$, and volume fraction, $\epsilon$ [@cooper2016$\tau$factor]. Tortuosity has been a metric of interest in a broad range of fields for a number of decades. In geophysics, $\tau$ influences groundwater flow through pourous rocks, which has significant environmental contamination impacts [@carey2016estimating]. Biomedical studies have used tortuosity as a diagnostic indicator of disease through imaging of retinal blood vessels [@hart1999measurement]. Electrochemists can use $\tau$ to model the diffusion of Li-ion through pourous electrodes, which can determine a cells power rating [@landesfeind2018tortuosity]. The imaging and subsequent modeling of materials to determine $\tau$ is thus commonplace.

\begin{equation}\label{eq:tort}
D_{eff} = D\dfrac{\epsilon}{\tau}
\end{equation}

![Flux field of a microstructure from the microlib library [@kench2022microlib].\label{fig:example}](example.pdf)

# Statement of need

Materials characterisation techniques are constantly improving, allowing the collection of larger field of view images with higher resolutions [@withers2021x]. Alongside these developments, machine learning algorithms have enabled the generation of arbitrarily large volumes, and can further enhance image quality through super-resolution techniques [@dahari2023fusion]. The resulting high fidelity microstructural datasets can be used to extract statistically representative metrics of a materials composition and performance. However, with increasingly dataset size, the computational cost to perform such analysis can lead to prohibitively long run times. This is especially true for intrisically 3D parameters such as the tortuosity factor, as the number of voxels scales cubically with sample edge length. Computational efficiency is also crucial for high-throughput tasks, for example during material optimisation. 

TauFactor 2 provides the necessary efficiency to ensure users can analyse large datasets in reasonable times. The software is built with pytorch, a commonly used and highly optimised python package for machine learning. The GPU acceleration that has enabled the drastic speed up of neural network training proves equally effective for the task of solving Laplacian equations, where matrix multiplication and addition are the central operations required. The use of python and pytorch ensure broad support and easy installation. The ability to run simulations with just a few lines of code ensures accessibility for researchers from the diverse fields where this software may be of merit.

The python implementation is similar to the original TauFactor MatLab software [@cooper2016taufactor]
, taking advantage of efficiency gains such as the precalculation of prefactors and the use of over-relaxation. The only significant difference is that in TauFactor 2, flux is calculated for all voxels. This replaces an indexing system in TauFactor, which enabled only diffusive voxels to solved. We find that the speed of GPU indexing compared to matrix multiplication makes this trade-off worthwhile. To compare the performance of TauFactor 2 to other available software, a test volume with different tortuosities in each direction has first been defined. This microstructure is available in the github repo, providing a standard against which new software can also be measured. The speed of four different solvers on a STATION SPECS, namely Taufactor2, Taufactor [@cooper2016taufactor]
, PoreSpy [@gostick2019porespy], Puma [@ferguson2018puma], are shown in figure \autoref{tauexample}. To check the accuracy of the calaculated tortuosities, we overconverge Taufactor to give a ‘true value’ in each direction. Using default convergence criteria, all five solvers are within 1% of the true values other then PuMa’s fast solver (5% error), which is thus excluded. Taufactor 2 is x times faster than the next best solver.

![Speed comparison for the four solvers.\label{fig:speeds}](tau_speeds.png)


# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References

# baby-andross

baby-andross is a collection of scripts that compute properties about changing neuronal connectivity in developing mouse cerebellum. These calculations are based on reconstructions of two high-resolution, electron microscopy (EM) image volumes (see Resources section below).

## The Datasets
(1) Postnatal day 3 (P3): 180 um x 120 um x 50 um

(2) Postnatal day 7 (P7): 180 um x 120 um x 75 um

Both volumes have a resolution of 4 nm/px x 4 nm/px x 30 nm/image section.

## Preparing to run these scripts
A few source files used for these analyses contain large amounts of data, and need to be uncompressed before running scripts that depend on them. These files are as follows:
- data/cf_voxel_lists/190306_p7_fta_vx_lists_patched.json.gz (voxel lists used to compute how climbing fiber terminal arbors are broken by the edges of the P7 image volume)
- data/mc_connectivity_experiment/190205_p3_obs_vs_mc_conn_niter_100000.json.gz (results of Monte Carlo simulating random innervation of targets by climbing fiber branches at P3)
- data/mc_connectivity_experiment/190205_p7_obs_vs_mc_conn_niter_100000.json.gz (results of Monte Carlo simulating random innervation of targets by climbing fiber branches at P7).

## Resources
The aligned EM image volumes and various manual annotations used for this analysis can be found here:
https://bossdb.org/project/PREwilson2019

The paper in which these scripts were used is here:
https://www.biorxiv.org/content/10.1101/627299v1


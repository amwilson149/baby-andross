# baby-andross

baby-andross is a collection of scripts that compute properties about changing neuronal connectivity in developing mouse cerebellum. These calculations are based on reconstructions of two high-resolution, electron microscopy (EM) image volumes (see Resources section for more information).

## The Datasets
(1) Postnatal day 3 (P3): 190 um x 120 um x 50 um

(2) Postnatal day 7 (P7): 190 $\mu$m x 120 um x 75 um

Both volumes have a resolution of 4 nm/px x 4 nm/px x 30 nm/image section.

## Preparing to run these scripts
One particular data file is large and needs to be uncompressed before running scripts that depend on it:
- data/cf_voxel_lists/190306_p7_fta_vx_lists_patched.json.gz (used by 190306_sim_cfs_from_ftas_p7_NEW.py)
Two other data files that are produced by these scripts are large. If you would like to inspect these results, compressed versions have been added to this repository:
- data/mc_connectivity_experiment/190205_p3_obs_vs_mc_conn_niter_100000.json.gz (produced by 190203_get_p3_syn_info_from_spreadsheet_axon_cuts.ipynb)
- data/mc_connectivity_experiment/190205_p7_obs_vs_mc_conn_niter_100000.json.gz (produced by 190205_get_p7_syn_info_from_mat_file_axon_cuts.ipynb)

## Resources
The aligned EM image volumes and various manual annotations used for this analysis can be found here:
https://bossdb.org/project/PREwilson2019

The paper in which these scripts were used is here:
https://www.biorxiv.org/content/10.1101/627299v1


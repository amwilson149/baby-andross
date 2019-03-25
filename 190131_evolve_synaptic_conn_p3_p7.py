import numpy as np
import scipy.stats as st
import scipy.io as scio
import json

# This code takes p3 connectivity and through stochastic synapse
# addition and removal attempt to evolve it into p7 connectivity

# Get connectivity matrices
# p3
p3cd = scio.loadmat('./181227_analysis_conn-based_ax_cuts/connectivity_matrices/P3_Observed_PC_Connectivity_Synapse_Numbers_gteq_5_syns_gteq_40pc_PC_targets.mat')
# p7
p7cd = scio.loadmat('./181227_analysis_conn-based_ax_cuts/connectivity_matrices/P7_Observed_PC_Connectivity_Synapse_Numbers_gteq_5_syns_gteq_70pc_PC_targets.mat')

p3c = p3cd['P3_PCconnectivity']
p7c = p7cd['P7_PCconnectivity']

# Define parameters to scan for simulation
# niter = 10 # number of times to simulate each parameter pair # niter for parameter scans
niter = 10 # number of times to simulate each parameter pair # niter for exploring dynamics
# prem = [0,0.01,0.05,0.1,0.5,1.0] # coarse first pass
# prem = np.arange(0,105,5)*0.001 # fine second pass
prem = [0.005] # long run to determine dynamics for a set of good parameters
# gamma = np.arange(-15,30,5)*0.1 # coarse first pass
# gamma = np.arange(0,205,5)*0.01 # fine second pass
gamma = [1.1] # long run to determine dynamics for a set of good parameters

# Set simulation flags
SCAN_PARAMS = 0 # if set to 1, each iteration of the simulation exits after the first time convergence occurs
TRACK_DYNAMICS = 0 # if set to 1 (should be for a single parameter pair, but any number of iterations)
# simulation will track errors and p-values per time step
TRACK_CONV_DYNAMICS = 0 # if set to 1 (should be for a single parameter pair, but any number of iterations)
# simulation will track errors and p-values per convergent time step
COMPUTE_SYN_RATES = 1 # if set to 1 (should be for a single parameter pair, but any number of iterations)
# simulation will track the number of synapses added per time step per cf and compute the average rate of
# synapse addition or removal per cf (with a linear regression fit of number of synapses per time step)
VERBOSE = 0 # if set to 1, comments stepping through the process will be displayed on the terminal

# Initialize lists to keep track of the fraction of times convergence
# occurred and the first time step at which convergence occurred
# (These lists will be combined and saved as JSON dictionaries)
# Information for all individual trials
ind_trials_prem = []
ind_trials_gamma = []
ind_trials_convergence = []
ind_trials_first_conv_ts = []
# Information averaged over the niter trials
avg_over_trials_prem = []
avg_over_trials_gamma = []
avg_over_trials_conv = []
avg_over_trials_first_conv_ts = []
# Information about the errors over time
err_ncfs_p_ts = []
err_npcs_p_ts = []
p_nsyns_p_ts = []
p_nsynspcf_p_ts = []
p_npcspcf_p_ts = []
p_ncfsppc_p_ts = []
err_ts = []
err_iters = []
err_info_dict = {}
# Information about the errors during convergent time steps
c_err_ncfs_p_ts = []
c_err_npcs_p_ts = []
c_p_nsyns_p_ts = []
c_p_nsynspcf_p_ts = []
c_p_npcspcf_p_ts = []
c_p_ncfsppc_p_ts = []
c_err_ts = []
c_err_iters = []
c_err_info_dict = {}
# Information about the total number of synapses per cf per unit time
id_convergent_trial = [] # only save the info if the trial starts converging at some point
# that way you can keep track of the fraction of convergent runs, too
cf_ids_by_ts = []
iter_curr_for_cfs_by_ts = []
ts_for_cfs_by_ts = []
n_syns_per_cf_by_ts = []
n_syns_per_cf_dict = {}

# Define functions for steps of simulation
def get_binary_matrix(c):
    nzr,nzc = np.where(c != 0)
    cbinary0 = np.zeros((c.shape[0],c.shape[1]))
    for q in range(len(nzr)):
        cbinary0[nzr[q],nzc[q]] = 1
    cbinary0.astype(int)
    return cbinary0

def initialize_conn(ci):
    # ci: the initial connectivity matrix
    c0 = ci
    cnz0 = np.asarray([q for q in ci.flatten() if q != 0])
    ncfs0 = ci.shape[0]
    npcs0 = ci.shape[1]
    nsynspcf0 = np.sum(ci,axis=1)
    cbinary0 = get_binary_matrix(ci)
    npcspcf0 = np.sum(cbinary0,axis=1)
    ncfsppc0 = np.sum(cbinary0,axis=0)
    return c0,cnz0,ncfs0,npcs0,nsynspcf0,npcspcf0,ncfsppc0

def remove_synapses(c,prem):
    # Chooses cf-pc pairs that will lose 1 synapse and decrements them
    # probability is "rate of synapse removal" (interpreted like a
    # c: the connectivity matrix that you're removing synapses from
    # prem: the base, per-synapse removal rate

    # Construct weighted removal probabilities for all cfs
    weights_rem_rate = np.tile(np.asarray(prem),(c.shape[0],c.shape[1]))
    weights_cf_syns = np.tile(np.expand_dims(np.asarray([q/np.sum(c) for q in np.sum(c,axis=1)]),axis=1),(1,c.shape[1]))
    bin_c = get_binary_matrix(c)
    weights_pc_uni = np.divide(bin_c , np.tile( np.expand_dims(np.sum(bin_c,axis=1),axis=1) , (1,c.shape[1]) ) )
    probs = np.multiply(weights_rem_rate, weights_cf_syns)
    probs = np.multiply(probs, weights_pc_uni)

    # Determine which cfs have synapse removal by drawing pseudo-
    # random numbers and identifying those less than the probabilities
    selectors = np.random.random_sample((c.shape[0],c.shape[1]))
    cnew = c
    for rid in range(c.shape[0]):
        for cid in range(c.shape[1]):
            elcurr = cnew[rid,cid]
            probcurr = probs[rid,cid]
            selcurr = selectors[rid,cid]
            if selcurr < probcurr:
                if VERBOSE == 1:
                    print('removing 1 synapse at row {0}, col {1}'.format(rid,cid))
                cnew[rid,cid] = elcurr - 1
    cnew.astype(int)
    return cnew

def add_synapses(c,gamma):
    # Chooses cf-pc pairs that will add 1 synapse and increments them
    # c is the connectivity matrix
    # gamma is the power that the number of synapses at a purkinje target is raised to
    # padd is the base synapse addition rate
    padd = 1
    weights_add_rate = np.tile(np.asarray(padd),(c.shape[0],c.shape[1]))
    weights_cf_syns = np.tile( np.expand_dims(np.asarray([q/np.sum(c) for q in np.sum(c,axis=1)]),axis=1) , (1,c.shape[1]) )
    c_to_gamma = np.zeros((c.shape[0],c.shape[1]))
    for rid in range(c.shape[0]):
        for cid in range(c.shape[1]):
            # all zero elements have no connection and should have weights equal to zeros here
            if c[rid,cid] != 0:
                c_to_gamma[rid,cid] = np.power(c[rid,cid].astype(float),gamma)
    norms_per_cf = np.tile( np.expand_dims(np.sum(c_to_gamma,axis=1),axis=1) , (1,c_to_gamma.shape[1]) )
    weights_pc_gamma = np.divide( c_to_gamma , norms_per_cf )
    probs = np.multiply(weights_add_rate, weights_cf_syns)
    probs = np.multiply(probs, weights_pc_gamma)
    # Determine which cfs have synapse removal by drawing pseudo-
    # random numbers and identifying those less than the probabilities
    selectors = np.random.random_sample((c.shape[0],c.shape[1]))
    cnew = c
    for rid in range(c.shape[0]):
        for cid in range(c.shape[1]):
            elcurr = cnew[rid,cid]
            probcurr = probs[rid,cid]
            selcurr = selectors[rid,cid]
            if selcurr < probcurr:
                if VERBOSE == 1:
                    print('adding 1 synapse at row {0}, col {1}'.format(rid,cid))
                cnew[rid,cid] = elcurr + 1
    cnew.astype(int)
    return cnew

def clear_disconnected_partners(c,row_labels,col_labels):
    rowsums = np.expand_dims(np.sum(c,axis = 1),axis=1)
    colsums = np.expand_dims(np.sum(c,axis = 0),axis=0)
    zero_rows_to_clear,dummy_cols = np.where(rowsums == 0)
    dummy_rows,zero_cols_to_clear = np.where(colsums == 0)
    fully_recon_col = np.expand_dims(c[:,0],axis=1)
    rows_disconn_from_full_recon_pc,dummy_cols = np.where(fully_recon_col == 0)
    all_rows_to_clear = list(set(zero_rows_to_clear.tolist() + rows_disconn_from_full_recon_pc.tolist()))
    all_cols_to_clear = list(set(zero_cols_to_clear.tolist()))
    cnew = c
    row_labels_new = row_labels
    col_labels_new = col_labels
    # remove rows to clear
    cnew = np.delete(cnew,all_rows_to_clear,axis=0)
    row_labels_new = np.delete(row_labels_new,all_rows_to_clear)
    # remove columns to clear
    cnew = np.delete(cnew,all_cols_to_clear,axis=1)
    col_labels_new = np.delete(col_labels_new,all_cols_to_clear)
    return cnew,row_labels_new,col_labels_new

def determine_if_converged(err_ncfs,err_npcs,p_nsyns,p_nsyns_pcf,p_npcspcf,p_ncfsppc,epsilon_ncfs,epsilon_npcs,alpha):
    has_converged = False
    # Simulation has converged if errors are less than epsilons and if p-values
    # are greater than alpha
    if (
            (err_ncfs <= epsilon_ncfs) \
        and (err_npcs <= epsilon_npcs)
        and (p_nsyns >= alpha) \
        and (p_nsyns_pcf >= alpha) \
        and (p_npcspcf >= alpha) \
        and (p_ncfsppc >= alpha)
    ):
        has_converged = True
    return has_converged


# Compute connectivity properties at p3 and p7
c_i,cnz_i,ncfs_i,npcs_i,nsynspcf_i,npcspcf_i,ncfsppc_i = initialize_conn(p3c)
c_f,cnz_f,ncfs_f,npcs_f,nsynspcf_f,npcspcf_f,ncfsppc_f = initialize_conn(p7c)

# Set maximum allowed errors in matrix size
# NOTE: we have only two data points for the ncfs at one Purkinje cell
# (one at p3, one at p7), and the same is true for the npcs innervated
# by one cf, and we independently determined that the numbers we observed
# are not statistically distinguishable
# So we choose the weak constraint that the ncfs and npcs are statistically
# indistinguishable as long as they are no further apart than the p3 and p7
# values (although if we have more observations they might still be statistically
# indistinguishable if they are farther apart).
epsilon_ncfs = np.abs(ncfs_f - ncfs_i)
epsilon_npcs = np.abs(npcs_f - npcs_i)

# Set alpha for determining when different distributions are
# statistically indistinguishable
alpha = 0.05

# Compute initial errors and p-values (i.e. those between p3 and p7 connectivity)
err_ncfs_i = np.abs(ncfs_f - ncfs_i)
err_npcs_i = np.abs(npcs_f - npcs_i)
p_nsyns_i = st.ranksums(cnz_i,cnz_f)[1]
p_nsynspcf_i = st.ranksums(nsynspcf_i,nsynspcf_f)[1]
p_npcspcf_i = st.ranksums(npcspcf_i,npcspcf_f)[1]
p_ncfsppc_i = st.ranksums(ncfsppc_i,ncfsppc_f)[1]

# Set bounds for time stepping
tmax = 5000

# Step into simulation loop
for pr in prem:

    for g in gamma:
        print('Running simulation with p_rem = {0} and gamma = {1}'.format(pr,g))

        # Reset variables that keep track of how often a parameter pair converges
        # and the mean time step at which initial convergence occurs
        frac_iter_conv = 0
        mean_ts_first_conv = 0

        for itercurr in range(niter):
            print('Running iteration {0}'.format(itercurr))

            # Reset convergence flags
            has_converged_curr = False
            has_converged_prev = False

            # Initialize connectivity to p3 values
            c_curr = c_i
            cnz_curr = cnz_i
            ncfs_curr = ncfs_i
            npcs_curr = npcs_i
            nsynspcf_curr = nsynspcf_i
            npcspcf_curr = npcspcf_i
            ncfsppc_curr = ncfsppc_i

            if VERBOSE == 1:
                print('shape of c_curr: {0}'.format(c_curr.shape))

            # Initialize a list of cf row ids so you can
            # keep track of individual cf connectivity even
            # as rows are removed
            cf_row_labels_i = np.arange(c_i.shape[0])
            cf_row_labels_curr = cf_row_labels_i
            # Do the same with pc column ids
            pc_col_labels_i = np.arange(c_i.shape[1])
            pc_col_labels_curr = pc_col_labels_i

            # Initialize errors and p-values
            err_ncfs_curr = err_ncfs_i
            err_npcs_curr = err_npcs_i
            p_nsyns_curr = p_nsyns_i
            p_nsynspcf_curr = p_nsynspcf_i
            p_npcspcf_curr = p_npcspcf_i
            p_ncfsppc_curr = p_ncfsppc_i

            # Start time stepping
            t = 0

            if VERBOSE == 1:
                print('STARTING TIME STEPPING')

            while ((t <= tmax) and not(has_converged_curr == False and has_converged_prev == True)):
                # Update lists of errors and p-values
                err_ncfs_p_ts.append(err_ncfs_curr)
                err_npcs_p_ts.append(err_npcs_curr)
                p_nsyns_p_ts.append(p_nsyns_curr)
                p_nsynspcf_p_ts.append(p_nsynspcf_curr)
                p_npcspcf_p_ts.append(p_npcspcf_curr)
                p_ncfsppc_p_ts.append(p_ncfsppc_curr)
                err_ts.append(t)
                err_iters.append(itercurr)

                # Update has_converged_prev
                has_converged_prev = has_converged_curr

                if has_converged_curr == None:
                    print('has_converged_curr has been set to none type at t = {0}. breaking.'.format(t))
                    break

                if VERBOSE == 1:
                    print('at time step t = {0}, the shape of c_curr is {1}.'.format(t,c_curr.shape)) # debugging
                # Choose cfs that will remove synapses and update
                # the connectivity matrix accordingly
                c_curr = remove_synapses(c_curr, pr)

                # Remove disconnected cfs (zero rows), disconnected
                # pcs (zero columns), and cfs that have become dis-
                # connected from the simulated "fully reconstructed pc"
                # (rows with zero in the first column)
                c_curr,cf_row_labels_curr,pc_col_labels_curr = clear_disconnected_partners(c_curr,cf_row_labels_curr,pc_col_labels_curr)
                # Choose cfs that will add synapses and update the
                # connectivity matrix
                c_curr = add_synapses(c_curr, g)

                # Compute updated connectivity properties for this matrix
                # and compare against the target matrix properties
                c_curr,cnz_curr,ncfs_curr,npcs_curr,nsynspcf_curr,npcspcf_curr,ncfsppc_curr = initialize_conn(c_curr)
                err_ncfs_curr = np.abs(ncfs_f - ncfs_curr)
                err_npcs_curr = np.abs(npcs_f - npcs_curr)
                p_nsyns_curr = st.ranksums(cnz_curr,cnz_f)[1]
                p_nsynspcf_curr = st.ranksums(nsynspcf_curr,nsynspcf_f)[1]
                p_npcspcf_curr = st.ranksums(npcspcf_curr,npcspcf_f)[1]
                p_ncfsppc_curr = st.ranksums(ncfsppc_curr,ncfsppc_f)[1]

                if COMPUTE_SYN_RATES:
                    cf_ids_by_ts.extend([int(cf_row_labels_curr[i]) for i in range(len(cf_row_labels_curr))])
                    iter_curr_for_cfs_by_ts.extend([itercurr for i in range(len(cf_row_labels_curr))])
                    ts_for_cfs_by_ts.extend([t for i in range(len(cf_row_labels_curr))])
                    n_syns_per_cf_by_ts.extend([int(nsynspcf_curr[i]) for i in range(len(nsynspcf_curr))])

                # debugging
                # print('errs: {0}, {1} \t p_vals: {2}, {3}, {4}, {5},'.format(err_ncfs_curr,err_npcs_curr,p_nsyns_curr,p_nsynspcf_curr,p_npcspcf_curr,p_ncfsppc_curr))

                if VERBOSE == 1:
                    print('check whether solution has converged:')
                has_converged_curr = determine_if_converged(err_ncfs_curr,err_npcs_curr,p_nsyns_curr,p_nsynspcf_curr,p_npcspcf_curr,p_ncfsppc_curr,epsilon_ncfs,epsilon_npcs,alpha)

                if VERBOSE == 1:
                    print(has_converged_curr,'\n')

                # Check for the occurrence of special conditions (convergence or destruction of the connectivity matrix)
                if SCAN_PARAMS:
                    if (has_converged_curr == True and has_converged_prev == False):
                        print('Simulation has begun to converge at time step {0}'.format(t))
                        # Add to lists that tell whether a simulation has converged and what time step it got to
                        ind_trials_prem.append(pr)
                        ind_trials_gamma.append(g)
                        ind_trials_convergence.append(has_converged_curr)
                        ind_trials_first_conv_ts.append(t)
                        # Update numbers used to compute fractions of convergence and mean time step of first convergence
                        frac_iter_conv += 1
                        mean_ts_first_conv += t
                        # stop simulation here
                        t=tmax
                if ((c_curr.shape[0] == 0) or (c_curr.shape[0] == 0)): # all cfs and pcs have been removed
                    print('All cfs and pcs have been removed from the connectivity matrix.')
                    t=tmax

                if TRACK_CONV_DYNAMICS:
                    if (has_converged_curr == True and has_converged_prev == False):
                        print('Simulation has begun to converge at time step {0}'.format(t))
                    if (has_converged_curr == True):
                        # Add information about the errors and p-values while convergence is occuring
                        c_err_ncfs_p_ts.append(err_ncfs_curr)
                        c_err_npcs_p_ts.append(err_npcs_curr)
                        c_p_nsyns_p_ts.append(p_nsyns_curr)
                        c_p_nsynspcf_p_ts.append(p_nsynspcf_curr)
                        c_p_npcspcf_p_ts.append(p_npcspcf_curr)
                        c_p_ncfsppc_p_ts.append(p_ncfsppc_curr)
                        c_err_ts.append(t)
                        c_err_iters.append(itercurr)

                if COMPUTE_SYN_RATES:
                    if (has_converged_curr == True and has_converged_prev == False):
                        print('Simulation has begun to converge at time step {0}'.format(t))
                        # stop simulation here
                        t=tmax

                t += 1

        if SCAN_PARAMS:
            # Add to lists that show averages over all iterations of when a parameter pair converged and at what initial time
            avg_over_trials_prem.append(pr)
            avg_over_trials_gamma.append(g)
            avg_over_trials_conv.append(frac_iter_conv/niter) # fraction of all iterations for which the parameter pair converged
            if frac_iter_conv > 0:
                avg_over_trials_first_conv_ts.append(mean_ts_first_conv/frac_iter_conv) # mean ts of first convergence FOR RUNS THAT CONVERGED
            else:
                avg_over_trials_first_conv_ts.append(-999) # we will have to take this number to mean that no trials ever converged

            # Save the single-iteration and averaged values into dictionaries that will be JSON serialized and saved
            ind_trial_results_dict = {'prem':ind_trials_prem,'gamma':ind_trials_gamma,'frac_conv_runs':ind_trials_convergence,'first_conv_ts':ind_trials_first_conv_ts}
            avg_trial_results_dict = {'prem':avg_over_trials_prem,'gamma':avg_over_trials_gamma,'frac_conv_runs':avg_over_trials_conv,'first_conv_ts':avg_over_trials_first_conv_ts}
            # For coarse runs
#             ind_fname = 'data/190205_evolve_p3_p7_coarse_scan_pr_{0}_{1}_g_{2}_{3}_ind_trial_conv_results.json'.format(prem[0],prem[-1],gamma[0],gamma[-1])
#             avg_fname = 'data/190205_evolve_p3_p7_coarse_scan_pr_{0}_{1}_g_{2}_{3}_avg_trial_conv_results.json'.format(prem[0],prem[-1],gamma[0],gamma[-1])
            # For fine runs
            ind_fname = 'data/190208_evolve_p3_p7_fine_scan_pr_{0}_{1}_g_{2}_{3}_ind_trial_conv_results.json'.format(prem[0],prem[-1],gamma[0],gamma[-1])
            avg_fname = 'data/190208_evolve_p3_p7_fine_scan_pr_{0}_{1}_g_{2}_{3}_avg_trial_conv_results.json'.format(prem[0],prem[-1],gamma[0],gamma[-1])
            with open(ind_fname,'w') as f:
                jsonobj = json.dumps(ind_trial_results_dict)
                f.write(jsonobj)
            with open(avg_fname,'w') as f:
                jsonobj = json.dumps(avg_trial_results_dict)
                f.write(jsonobj)

        if TRACK_DYNAMICS:
            # Generate dictionary containing errors and p-values during all time steps for all iterations
            err_info_dict['err_ncfs'] = [int(err_ncfs_p_ts[q]) for q in range(len(err_ncfs_p_ts))] # to make JSON serializable
            err_info_dict['err_npcs'] = [int(err_npcs_p_ts[q]) for q in range(len(err_npcs_p_ts))]
            err_info_dict['p_nsyns'] = [int(p_nsyns_p_ts[q]) for q in range(len(p_nsyns_p_ts))]
            err_info_dict['p_nsynspcf'] = [int(p_nsynspcf_p_ts[q]) for q in range(len(p_nsynspcf_p_ts))]
            err_info_dict['p_npcspcf'] = [int(p_npcspcf_p_ts[q]) for q in range(len(p_npcspcf_p_ts))]
            err_info_dict['p_ncfsppc'] = [int(p_ncfsppc_p_ts[q]) for q in range(len(p_ncfsppc_p_ts))]
            err_info_dict['p_ncfsppc'] = [int(p_ncfsppc_p_ts[q]) for q in range(len(p_ncfsppc_p_ts))]
            err_info_dict['time_step'] = [int(err_ts[q]) for q in range(len(err_ts))]
            err_info_dict['iteration'] = [int(err_iters[q]) for q in range(len(err_iters))]
            # Save dictionary
            errfname = './data/190211_errs_per_ts_pr_{0}_g_{1}_niter_{2}.json'.format(prem[0],gamma[0],niter)
            with open(errfname,'w') as f:
                jsonobj = json.dumps(err_info_dict)
                f.write(jsonobj)

        if TRACK_CONV_DYNAMICS:
            # Generate dictionary containing errors and p-values during convergent time steps for all iterations
            c_err_info_dict['err_ncfs'] = [int(c_err_ncfs_p_ts[q]) for q in range(len(c_err_ncfs_p_ts))]
            c_err_info_dict['err_npcs'] = [int(c_err_npcs_p_ts[q]) for q in range(len(c_err_npcs_p_ts))]
            c_err_info_dict['p_nsyns'] = [int(c_p_nsyns_p_ts[q]) for q in range(len(c_p_nsyns_p_ts))]
            c_err_info_dict['p_nsynspcf'] = [int(c_p_nsynspcf_p_ts[q]) for q in range(len(c_p_nsynspcf_p_ts))]
            c_err_info_dict['p_npcspcf'] = [int(c_p_npcspcf_p_ts[q]) for q in range(len(c_p_npcspcf_p_ts))]
            c_err_info_dict['p_ncfsppc'] = [int(c_p_ncfsppc_p_ts[q]) for q in range(len(c_p_ncfsppc_p_ts))]
            c_err_info_dict['p_ncfsppc'] = [int(c_p_ncfsppc_p_ts[q]) for q in range(len(c_p_ncfsppc_p_ts))]
            c_err_info_dict['time_step'] = [int(c_err_ts[q]) for q in range(len(c_err_ts))]
            c_err_info_dict['iteration'] = [int(c_err_iters[q]) for q in range(len(c_err_iters))]
            # Save dictionary
            cerrfname = './data/190211_errs_per_conv_ts_pr_{0}_g_{1}_niter_{2}.json'.format(prem[0],gamma[0],niter)
            with open(cerrfname,'w') as f:
                jsonobj = json.dumps(c_err_info_dict)
                f.write(jsonobj)

        if COMPUTE_SYN_RATES:
            # Generate dictionary containing number of synapses per cf per time step
            n_syns_per_cf_dict['cf_id'] = cf_ids_by_ts
            n_syns_per_cf_dict['sim_iteration'] = iter_curr_for_cfs_by_ts
            n_syns_per_cf_dict['timestep'] = ts_for_cfs_by_ts
            n_syns_per_cf_dict['nsyns_per_ts'] = n_syns_per_cf_by_ts
            # Save dictionary
            nsyns_fname = 'data/190211_evolve_p3_p7_nsyns_per_cf_per_ts_pr_{0}_g_{1}_niter_{2}.json'.format(prem[0],gamma[0],niter)
            with open(nsyns_fname,'w') as f:
                jsonobj = json.dumps(n_syns_per_cf_dict)
                f.write(jsonobj)

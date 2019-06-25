import numpy as np
import random
import scipy.ndimage.measurements as spim
import skimage.measure as skim
import json

# List of cfs with full terminal arbors
p7_ftas = [7,11,12,22,24,54,55,61,63] # p7

# all cf ids from p7 analysis for reference
# p7_cfs_1 = [2,3,4, 5, 6, 7, 9,11,12,13,14,15,16,19,20,22,23,24,29,30,31,33,34,36,37,39,40,41,42,44,45,46,49,50,51,54,55,56,59,60,61,63]
# p7_cfs_2 = [1,2,3,4,5,9] # axons [64,65,66,67,68,72], traced in "P7_Put_CF_and_MF_Extensions_2-9-2016.vsseg"
# p7_cfs_3 = [2] # axon [81], traced in "P7_PC1_Axon_Extension_Inputs.vsseg"

# Dataset resolutions of interest
p7_res_1 = [128, 128, 30]
mip_3_res = [32, 32, 30]

# Import p3 and p7 voxel lists (which were computed separately because they take so long)
print('Importing voxel lists...')
p7_vx_fname = './data/190306_p7_fta_vx_lists_patched.json'
with open(p7_vx_fname,'r') as f:
    p7_vx = json.loads(f.read())

# Define approximate full volume sizes at the mip sizes at which the voxel
# lists for each were generated (p3: , p7: ), in voxels
# The sizes below are written as the MIP3 values divided by the downsampling
# factor that gets you to the voxel list MIP level
p7ds = p7_res_1[0]/mip_3_res[0]
print(p7ds) # debugging
p7xsize = 3750/p7ds # 120 um
p7ysize = 5938/p7ds # 190 um
p7zsize = 2514

# # Set limits on translations and rotations for data augmentation
# Translations
p7_xmin = -p7xsize
p7_ymin = -0.01*p7ysize
p7_zmin = -p7zsize
p7_xmax = p7xsize
p7_ymax = 0.01*p7ysize
p7_zmax = p7zsize
# Rotations: alpha, beta, and gamma are rotation angles about x, y, and z
# CARTESIAN axes (see below)
p7_alphamin = 0#-np.pi/4.0
p7_betamin = 0
p7_gammamin = 0#-np.pi/4.0
p7_alphamax = 0#np.pi/4.0
p7_betamax = 0#2*np.pi
p7_gammamax = 0#np.pi/4.0

# Check values
print('Min and max translation and rotation parameters:')
print('p7:')
print(p7_xmin,p7_xmax,p7_ymin,p7_ymax,p7_zmin,p7_zmax,p7_alphamin,p7_alphamax,p7_betamin,p7_betamax,p7_gammamin,p7_gammamax)

# # Define the sample sizes to generate, and the number of times
# # to do so per sample size
N0 = [1] # sample size
n_iter = 500 # number of times to generate samples of size N

# Define rotations about x, y, and z axes in the upward-pointing Cartesian coord. system
# For our images, z is -x, x is y, and y is z
# Quick reference for image coords to Cartesian:
# z -> -x
# x -> y
# y -> z
# The coords. of our images still follow the right-hand rule,
# so relatively speaking these rotations should still be fine.
# Just have to be careful to pick appropriate angles.
def rx(pos,alpha):
    # pos is a 3x1 numpy-like array with image coordinate values
    # alpha is an angle in radians
    x0 = pos[0]
    y0 = pos[1]
    z0 = pos[2]
    x = x0
    y = np.cos(alpha)*y0 - np.sin(alpha)*z0
    z = np.sin(alpha)*y0 + np.cos(alpha)*z0
    return [x,y,z]

def ry(pos,beta):
    # pos is a 3x1 numpy-like array with image coordinate values
    # beta is an angle in radians
    x0 = pos[0]
    y0 = pos[1]
    z0 = pos[2]
    x = np.cos(beta)*x0 + np.sin(beta)*z0
    y = y0
    z = -np.sin(beta)*x0 + np.cos(beta)*z0
    return [x,y,z]

def rz(pos,gamma):
    # pos is a 3x1 numpy-like array with image coordinate values
    # gamma is an angle in radians
    x0 = pos[0]
    y0 = pos[1]
    z0 = pos[2]
    x = np.cos(gamma)*x0 - np.sin(gamma)*y0
    y = np.sin(gamma)*x0 + np.cos(gamma)*y0
    z = z0
    return [x,y,z]

# Define a function to generate an augmented set of cfs
# l: lower bound; u: upper bound
def augment_cf_sample(cfvx,xvolsize,zvolsize,yl,yu,alphl,alphu,betl,betu,gaml,gamu,ssize=1):
    # cfvx is a dictionary with cf ids stored as 'seg_id' and an array with a voxel list for each
    # stored as 'vox_lists'
    print('Creating sample of size {0}'.format(ssize))
    randintmax = len(cfvx['seg_id'])
    sample = []
    id_choices = []
    transforms = []
    for i in range(ssize):
        # Choose cf
        cfid = random.randint(0,randintmax-1) # bounds are inclusive for randint
        id_choices.append(cfvx['seg_id'][cfid])

        # Get voxel list for current cf fta
        vx = cfvx['vox_lists'][cfid] # this should be list-of-lists format now

        # NOTE ABOUT FUTURE IMPROVEMENT
        # Rotate climbing fiber fta by amount chosen about the y axis only
        # Rotations about x and z axes need to be about the x and z axes and
        # they're not right now, so have to fix later if you want them to be
        # Also perform rotations prior to translations so that you constrain the
        # movement appropriately

        # Compute limits on translations along the x and z axes based on bbox for
        # current cf voxel list
        xvx = [q[0] for q in vx]
        zvx = [q[2] for q in vx]
        minx = np.min(xvx)
        maxx = np.max(xvx)
        minz = np.min(zvx)
        maxz = np.max(zvx)
        # Set constraints on translation in x and z so that all segments can at
        # maximum translation can be just outside the volume
        delxminus = maxx - 1
        delxplus = xvolsize-minx - 1
        delzminus = maxz - 1
        delzplus = zvolsize - minz - 1
        xl = -delxminus
        xu = delxplus
        zl = -delzminus
        zu = delzplus

        # Debugging (I believe this works properly atm)
        # print('volume limits:')
        # print('x = 0, {0}, z = 0, {1}'.format(xvolsize,zvolsize))
        # print('seg x-z bbox vals:')
        # print('x = {0}, {1}, z = {2}, {3}'.format(minx,maxx,minz,maxz))
        # print('translation limits:')
        # print('x = {0}, {1}, z = {2}, {3}'.format(xl,xu,zl,zu))
        # print('\n')

        # Set translation
        tx = random.uniform(xl,xu)
        ty = random.uniform(yl,yu)
        tz = random.uniform(zl,zu)
        # Set rotation
        alpha = random.uniform(alphl,alphu)
        beta = random.uniform(betl,betu)
        gamma = random.uniform(gaml,gamu)

        # # Debugging
        # print('translations:')
        # print(tx,ty,tz) # debugging
        # print('rotations:')
        # print(alpha,beta,gamma) # debugging
        # print('\n')

        # Apply rotations
        vxnew = [rx([q[0],q[1],q[2]],alpha) for q in vx] # x-rot
        vxnew = [ry([q[0],q[1],q[2]],beta) for q in vxnew] # y-rot
        vxnew = [rz([q[0],q[1],q[2]],gamma) for q in vxnew] # z-rot
        # Apply translations
        vxnew = [[np.int16(q[0]+tx),np.int16(q[1]+ty),np.int16(q[2]+tz)] for q in vxnew]

        xvxn = [q[0] for q in vxnew]
        zvxn = [q[2] for q in vxnew]
        # print('translated seg x-z bbox:')
        # print('x = {0}, {1}, z = {2}, {3}\n'.format(np.min(xvxn),np.max(xvxn),np.min(zvxn),np.max(zvxn)))

        # Add new voxel list to sample
        sample.append(vxnew)
        transforms.append([tx,ty,tz,alpha,beta,gamma])
    sample_dict = {'baseline_seg_id':id_choices,'vox_lists':sample}
    return sample_dict,transforms

# Define a function to compute the number of connected components in a voxel list
def get_conn_comp(vx):
    # vx is a list-of-list-type structure, with each element containing [vx,vy,vz]
    # Create binary 3d numpy array
    vxx = [q[0] for q in vx]
    vxy = [q[1] for q in vx]
    vxz = [q[2] for q in vx]
    x1 = np.min(vxx)
    x2 = np.max(vxx)
    y1 = np.min(vxy)
    y2 = np.max(vxy)
    z1 = np.min(vxz)
    z2 = np.max(vxz)
#     print(x1,x2,y1,y2,z1,z2)
    if ((x2 > x1) and (y2 > y1) and (z2 > z1) ):
        img = np.zeros([x2-x1+1,y2-y1+1,z2-z1+1])
        print(img.shape,x1,y1,z1,x2,y2,z2) # debugging
        for i in range(len(vxx)):
            # Remove offset from the coords so they fit in your image
            img[vxx[i]-x1,vxy[i]-y1,vxz[i]-z1]=1
        img = img.astype(np.uint16) # make sure the image is integer type
        structuring_element = np.ones([3,3,3]).astype(np.uint16) # voxels that are 26-connected are single regions #np.uint8
        # Determine number of connected components by labeling features
        # using the kernel structure above
        print(type(img))
        print(type(structuring_element))
        label,nccs = spim.label(img,structure = structuring_element,output=np.uint16)
    else:
        print('voxel list is corrupt')
        nccs = -999
    return nccs

# Compute the number of connected components in the baseline voxel lists per cf
# so you can correct for gaps etc. in the voxel lists before doing the transformation experiment
print('Computing baseline numbers of connected components...')
print('p7')
p7_ccs = []
for item in p7_vx['vox_lists']:
    bcc_curr = get_conn_comp(item)
    print('baseline number of conn. comp. for current segment = {0}'.format(bcc_curr))
    p7_ccs.append(bcc_curr)
p7_baseline_cc_dict = {'seg_id':p7_vx['seg_id'],'n_ccs_bl':p7_ccs}

# Create a collection of samples for each sample size, count the number of
# connected regions inside the image volume for the sample, repeat n_iter
# times, and store the results in a dictionary.
def runsim(cfvx,xl,xu,yl,yu,zl,zu,alphl,alphu,betl,betu,gaml,gamu,ssizes,baseline_cc_dict,volx,voly,volz,n_iter=10):
    cfidlists_allssizes = []
    nconncomp_allssizes = []
    for ncurr in ssizes:
        cfidlists = []
        nconncomp = []
        for iter in range(n_iter):
            print('Generating {0}th new sample of size {1}'.format(iter,ncurr))
            samplecurr,transcurr = augment_cf_sample(cfvx,volx,volz,yl,yu,alphl,alphu,betl,betu,gaml,gamu,ncurr)
            cfidlist = []
            conncompdist = []
            # For each sample, remove voxels outside the bounds of the volume
            # Then calc number of connected regions in the volume bounding the voxel list
            for s in range(len(samplecurr['baseline_seg_id'])):
                print('Counting connected components for item {0} in sample'.format(s))
                sicurr = int(samplecurr['baseline_seg_id'][s])
                cfidlist.append(sicurr) # re-assigning ids to a new list to make sure cfid order is preserved
                vxcurr = samplecurr['vox_lists'][s]
                vxkeep = []
                passcount = 0
                failcount = 0
                for i in range(len(vxcurr)):
                    q = vxcurr[i]
                    # print(q)
                    if (
                            (q[0] >= 0) \
                        and (q[0] < volx) \
                        and (q[1] >= 0) \
                        and (q[1] < voly) \
                        and (q[2] >= 0) \
                        and (q[2] < volz)
                        ):
                        passcount +=1
                        vxkeep.append([int(q[0]),int(q[1]),int(q[2])])
                    else:
                        failcount += 1
                print('{0} voxels in segment {1} fell inside the volume and {2} voxels were outside \n This segment initially had {3} voxels'.format(passcount,s,failcount,len(vxcurr)))
                if (len(vxkeep)) > 0:
                    cccurr = get_conn_comp(vxkeep)

                    # Subtract the baseline number of connected components above 1 for the root cf id
                    # (to correct for gaps)
#                     baseline_locs = [i for i,q in enumerate(baseline_cc_dict['seg_id']) if q == sicurr]
#                     print(baseline_locs) # debugging
                    # Don't want to make this "correction", actually, because things
                    # with m number of ccs normally in the volume might be far enough out that
                    # only a small part of it is in the volume and so even though there are new pieces
                    # formed by e.g. exit and re-entry in the volume, there are way less because the original
                    # segment isn't really in the volume at all.
                    # What you might want to do is to compare distributions, and then to try and fill
                    # holes in voxel lists so that they start out having only 1 connected component
#                     ccbl = baseline_cc_dict['n_ccs_bl'][baseline_locs[0]] # there should only be one element in this list
#                     print(ccbl) # debugging
#                     cc_correction = ccbl - 1
#                     cccurr = cccurr - cc_correction
#                     print(cccurr, cc_correction)
                else:
                    cccurr = 0
                conncompdist.append(cccurr)
            cfidlists.append(cfidlist)
            nconncomp.append(conncompdist)
        cfidlists_allssizes.append(cfidlists)
        nconncomp_allssizes.append(nconncomp)
    simdict = {'sample_size':ssizes,'cf_id_lists':cfidlists_allssizes,'conn_comp_dists':nconncomp_allssizes}
    return simdict

# Run the simulation for p7 data
print('Running simulation of P7 data...')
p7_simdict0 = runsim(p7_vx,p7_xmin,p7_xmax,p7_ymin,p7_ymax,p7_zmin,p7_zmax,p7_alphamin,p7_alphamax,p7_betamin,p7_betamax,p7_gammamin,p7_gammamax,N0,p7_baseline_cc_dict,p7xsize,p7ysize,p7zsize,n_iter)
print('Saving simulation results...')
# Save dictionary about number of connected components per terminal arbor
p7_fta_fname0 = 'data/sim_br_from_ftas_results/190307_new_dxdz_const_p7_n_segs_from_ftas_ssize_{0}_{1}_n_iter_{2}.json'.format(N0[0],N0[-1],n_iter)
with open(p7_fta_fname0,'w') as f:
    jsonobj = json.dumps(p7_simdict0)
    f.write(jsonobj)
# Save dictionary with the baseline numbers of connected components per terminal arbor
p7_bl_fname = 'data/sim_br_from_ftas_results/190307_new_dxdz_const_p7_baseline_n_ccs.json'
print('saving p7 baseline connected components...')
with open(p7_bl_fname, 'w') as f:
    jsonobj = json.dumps(p7_baseline_cc_dict)
    f.write(jsonobj)

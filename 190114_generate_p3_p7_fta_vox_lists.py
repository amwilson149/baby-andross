import numpy as np
import random
import scipy.ndimage.measurements as spim
import scipy.ndimage.filters as spif
import json

from cloudvolume import CloudVolume

# List of cfs with full terminal arbors
p3_ftas = [62,63,145,154,158,206,298,226] # p3
p7_ftas = [7,11,12,22,24,54,55,61,63] # p7

# Dataset resolutions of interest
p3_res = [64, 64, 30]
p7_res_1 = [128, 128, 30]

# Get voxel lists of these processes (at a higher MIP level for speed)
p3_vol = CloudVolume('gs://wilson-cerebellum/p3-cf-segs-1-mip3/cfs-mip3-1',mip=p3_res)
p7_vol_1 = CloudVolume('gs://wilson-cerebellum/p7-cf-segs-1-mip3/cfs-mip3-1',mip=p7_res_1) # all ftas are in seg layer 1


# Define a function to patch any holes in the initial skeletons you pull
def patch_holes(vox):
    xs = [q[0] for q in vox]
    ys = [q[1] for q in vox]
    zs = [q[2] for q in vox]
    imx = np.max(xs)
    imy = np.max(ys)
    imz = np.max(zs)
    print('proposed image size = ({0},{1},{2})'.format(imx,imy,imz))

    img = np.zeros((imx+1,imy+1,imz+1)) # so indices go from (0,0,0) to (imx,imy,imz)
    for q in range(len(xs)):
        img[xs[q],ys[q],zs[q]] = 1
    # Convolve image with a kernel of increasing size just until there is only
    # one connected component for that segment
    ksize = 3 # Start out with a 3x3x3 kernel
    oneconncomp = False
    while not oneconncomp:
        kernel = np.ones((ksize,ksize,ksize)).astype(np.uint16)
        img2 = spif.convolve(img,kernel)
        # Check the number of connected components in resulting voxel list
        se = np.ones([3,3,3]).astype(np.uint16)
        labels, nccs = spim.label(img2,structure=se,output=np.uint16)
        print('Number of connected components in voxel list = {0}'.format(nccs))
        if nccs == 1:
            oneconncomp = True
            vxn,vyn,vzn = np.where(img2)
        else:
            print('Incrementing conv. kernel size to produce 1 conn. comp.')
            ksize = ksize + 1
    voxnew = [[vxn[q],vyn[q],vzn[q]] for q in range(len(vxn))]
    return voxnew

# Get voxel lists by finding bbox for the corresponding skeleton (should be fast)
def get_vox_lists(ids,vol,res):
    ids_to_add = [] # added so this function works when you run a subset of input ids
    vox_lists = []
    for id in ids:
        print('processing id {0}'.format(id))
        skel = vol.skeleton.get(id)
        verts = skel.vertices # verts are in nm
        x1 = (np.min(verts[:,0])/res[0]).astype(np.uint32)
        x2 = (np.max(verts[:,0])/res[0]).astype(np.uint32)
        y1 = (np.min(verts[:,1])/res[1]).astype(np.uint32)
        y2 = (np.max(verts[:,1])/res[1]).astype(np.uint32)
        z1 = (np.min(verts[:,2])/res[2]).astype(np.uint32)
        z2 = (np.max(verts[:,2])/res[2]).astype(np.uint32)
#         print(x1,x2,y1,y2,z1,z2) # debugging
        svol = np.array(vol[x1:x2,y1:y2,z1:z2],dtype=np.uint32)
        vox = np.where(svol == id)
        # Add the offset for the volume bbox back to locs so
        # they are correct
        # Put voxel list in a form that is JSON serializable:
        # 1) rearrange voxel list so it's a list of lists
        # 2) convert to Python int
        vx0 = [q for q in vox[0]]
        vy0 = [q for q in vox[1]]
        vz0 = [q for q in vox[2]]
        vx = vx0+x1
        vy = vy0+y1
        vz = vz0+z1
#         print(np.min(vx),np.max(vx),np.min(vy),np.max(vy),np.min(vz),np.max(vz)) # debugging
        vox = [[int(vx[i]),int(vy[i]),int(vz[i])] for i in range(len(vx))]
        # Consistency check: make sure that the set of values inside the voxel
        # list are restricted to the id you were looking for
#         id_check = list(set([svol[vx0[i],vy0[i],vz0[i],0] for i in range(len(vx0))]))
#         print(id_check)
        # Convolve image of voxels with a 3-dimensional kernel to patch holes
        vox_patched = patch_holes(vox)
        vox_lists.append(vox_patched)
        ids_to_add.append(id)
    # Store voxel lists in a dictionary
    vx_dict = {'seg_id':ids_to_add, 'vox_lists':vox_lists}
    return vx_dict

print('Getting voxel lists...')
# p3_vx = get_vox_lists(p3_ftas,p3_vol,p3_res)
# p3_vx_fname = 'data/190306_p3_fta_vx_lists_patched.json'
# with open(p3_vx_fname,'w') as f:
#     jsonobj = json.dumps(p3_vx)
#     f.write(jsonobj)
# del p3_vx # delete this variable to avoid a memory issue later in the script

p7_vx = get_vox_lists(p7_ftas,p7_vol_1,p7_res_1)
p7_vx_fname = 'data/190306_p7_fta_vx_lists_patched.json'
with open(p7_vx_fname,'w') as f:
    jsonobj = json.dumps(p7_vx)
    f.write(jsonobj)

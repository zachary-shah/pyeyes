import numpy as np
from pyeyes.viewers import ComparativeViewer

# Data
mrf_folder = '/local_mount/space/mayday/data/users/zachs/zachplotlib/data/mrf'

llr_1min_pd = np.load(f'{mrf_folder}/llr_1min_pd.npy')
llr_1min_t1 = np.load(f'{mrf_folder}/llr_1min_t1.npy')
llr_1min_t2 = np.load(f'{mrf_folder}/llr_1min_t2.npy')
llr_2min_pd = np.load(f'{mrf_folder}/llr_2min_pd.npy')
llr_2min_t1 = np.load(f'{mrf_folder}/llr_2min_t1.npy')
llr_2min_t2 = np.load(f'{mrf_folder}/llr_2min_t2.npy')

# NOTE: future version will allow for categorical description along an axis. e.g. ['PD', 'T1', 'T2']
mrf_1min = np.stack([llr_1min_pd, llr_1min_t1, llr_1min_t2], axis=0)
mrf_2min = np.stack([llr_2min_pd, llr_2min_t1, llr_2min_t2], axis=0)

img_dict = {
    '1min': mrf_1min,
    '2min': mrf_2min
}

# Parameters
named_dims = ['MRF Type', 'x', 'y', 'z']
vdims = ['y', 'z']
cat_dims = {'MRF Type': ['PD', 'T1', 'T2']}

Viewer = ComparativeViewer(
    data = img_dict,
    named_dims = named_dims,
    view_dims = vdims,
    cat_dims=cat_dims,
)

Viewer.launch()
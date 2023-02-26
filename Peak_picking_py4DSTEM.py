## Want to make automated orientation map using py4D STEM and pyxem- 4D STEM to pick peaks and simulate many diffraction patterns in the steroegraphic traingle- dones't matter about in plane rotation only z- then do this for all diffraction patterns and compare for multiple .cif files
## Import
import pyxem as pxm
import py4DSTEM
import numpy as np
import hyperspy.api as hs
import h5py
import dask.array as da
import pickle
import os,glob
import matplotlib.pyplot as plt
print(py4DSTEM.__version__)

## Firstly load the data using py4DSTEM
file_path_data = r"D:\dg606\SED\Centered_and_aff_trans\20221202_183724.hdf5"
file_path_probe = r"C:\Users\tas72\Downloads\20220811_211451-20230131T093145Z-001\20220811_211451\binned_diff_20220811_211451.hdf5"
file_path_analysis = r"C:\Users\tas72\Documents\PhD\dg606\py4DSTEM_analysis\20221202_183724/"

if not os.path.exists(file_path_analysis):
    os.makedirs(file_path_analysis)
    print("Experiment analysis folder created!")
    
# From calibration using pyxem (see hyperspy value)
inv_Ang_per_pixel = 0.01053034801747163
nm_per_nav_pixel = 2.007876


with h5py.File(file_path_data,'r') as f:
  dataset = da.from_array(f['Experiments/__unnamed__/data'])
  dataset = dataset.compute()
  dataset = py4DSTEM.io.DataCube(dataset)

with h5py.File(file_path_probe,'r') as f:
  dataset_probe = da.from_array(f['Experiments/__unnamed__/data'])
  dataset_probe = dataset_probe[:20,:20,:,:]
  dataset_probe = dataset_probe.compute()
  dataset_probe = py4DSTEM.io.DataCube(dataset_probe)
    
## Just as the probe wa from a differnet measurement- works okay with this though
dataset_probe.data = dataset_probe.data[:,:,:-1,:-1]

# Nav axis calibration
dataset.calibration.set_R_pixel_size(nm_per_nav_pixel)
dataset.calibration.set_R_pixel_units('nm')

# Get max and 
dataset.get_dp_max(verbose=True)
dataset.get_dp_mean(verbose=True)
dataset_probe.get_dp_mean(verbose=True)


## Plot mean and Max diff Patterns
titles = ['dp_mean','dp_max','dp_mean_probe']
fig,ax = plt.subplots(1,3)
ax[0].imshow(dataset.tree['dp_mean'].data,vmax=10)
ax[1].imshow(dataset.tree['dp_max'].data,vmax=100)
ax[2].imshow(dataset_probe.tree['dp_mean'].data,vmax=0.1)
for i in range(3):
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_title(titles[i])
plt.savefig(file_path_analysis+'mean_and_max_dps.png',dpi =300)
print('Mean and Max dps made')
## Make and plot a VDF
from py4DSTEM.process.virtualimage import get_virtual_image

center = (dataset.Qshape[0]//2 , dataset.Qshape[1]//2)
radii = (8, 130)



# Calculate the ADF image
dataset.get_virtual_image(
    mode = 'annulus',
    geometry = (center,radii),
    name = 'dark_field',
)
# Plot
fig,ax = plt.subplots()
ax.imshow(dataset.tree['dark_field'].data,cmap ='gray')
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(file_path_analysis+'vdf.png',dpi=300)
print('VDF made')

## Construct probe template
probe_align = py4DSTEM.process.probe.get_vacuum_probe(
    dataset_probe,
    # Only take the first 10x10 pixels
    ROI = (0,10,0,10),
    mask_threshold = 0.02,
    align = True,
)

probe_semiangle, probe_qx0, probe_qy0 = py4DSTEM.process.calibration.get_probe_size(
    probe_align,
    thresh_lower = 0.02,
    thresh_upper = 0.04,
)

print('Estimated probe radius =', '%.2f' % probe_semiangle, 'pixels')

probe_kernel = py4DSTEM.process.probe.get_probe_kernel_edge_sigmoid(
    probe_align,
    (probe_semiangle * 0.0, probe_semiangle * 3.0),
#     bilinear=True,
)

print('Probe formed')

## Disk detection
# Hyperparameters
detect_params = {
    'corrPower': 1,
    'sigma': 3,
    'edgeBoundary': 8,
    'minRelativeIntensity': 0.001,
    'minAbsoluteIntensity': 3,
    'minPeakSpacing': 10,
    'subpixel' : 'poly',
#     'subpixel' : 'multicorr',
    'upsample_factor': 8,
    'maxNumPeaks': 30,
#     'CUDA': True,
}
#Sample spots 
rxs = 311, 463, 78, 462, 297, 339
rys = 109, 269, 267, 36, 276, 385
colors=['r','limegreen','c','g','orange', 'violet']
disks_selected = dataset.find_Bragg_disks(
    data = (rxs, rys),
    template = probe_kernel,
    **detect_params,
)

fig = py4DSTEM.visualize.show_image_grid(
    get_ar = lambda i:dataset.data[rxs[i],rys[i],:,:],
    H=2, 
    W=3,
    axsize=(5,5),
    clipvals='manual',
    vmin=0,
    vmax=30,
    scaling='power',
    power=0.3,
    get_bordercolor = lambda i:colors[i],
    get_x = lambda i: disks_selected[i].data['qx'],
    get_y = lambda i: disks_selected[i].data['qy'],
    get_pointcolors = lambda i: colors[i],
    open_circles = True,
    scale = 400,
    returnfig=True
)


plt.savefig(file_path_analysis+'_disk_detection.png',dpi=300)

## Find all b peaks
bragg_peaks = dataset.find_Bragg_disks(
    template = probe_kernel,
    **detect_params,
)

## Save them
# Save Bragg disk positions
file_name_braggdisks_raw = file_path_analysis+'_braggdisks_raw.h5'
py4DSTEM.save(
    file_name_braggdisks_raw,
    bragg_peaks,
    mode='o',
)
print('Bragg peaks picked and saved')

###### Here is where you would apply a mask if needed- we dont require this as single chip detector ######

mask = np.zeros((256,256))
# hot_pixels = [[154,72],[230,120],[165,101],[234,244]]

# for pix in hot_pixels:
#     mask[pix[1]-1:pix[1]+1,pix[0]-1:pix[0]+1] = 1


# # Circular
# r_to_mask = 20 #px
# s = dataset.Qshape
# xx, yy = np.mgrid[:s[0], :s[1]]
# circle = (xx - 128) ** 2 + (yy - 128) ** 2
# mask = np.logical_and(circle < (r_to_mask **2), circle >= 0 **2) # second number = inner radius

# # Cross
# w = 4 #px
# x, y = 120, 153 #px
# mask = np.zeros(shape = dataset.Qshape)
# mask[x - w//2 : x + w//2, :] = 1
# mask[:, y - w//2 : y + w//2] = 1


# Apply the mask to the raw bragg peaks
bragg_peaks_masked = bragg_peaks.get_masked_peaks(
    mask,
)

# Create a bragg vector map (2D histogram of all detected bragg peaks) for both raw and masked Bragg peaks
bragg_vector_map = bragg_peaks.get_bvm(mode='raw')
bragg_vector_map_masked = bragg_peaks_masked.get_bvm(mode='raw')


# Save masked Bragg peaks
file_name_braggdisks_masked = file_path_analysis+'braggdisks_masked.h5'
py4DSTEM.save(
    file_name_braggdisks_masked,
    bragg_peaks_masked,
    mode='o',
)
fig,ax = plt.subplots(1,2)
ax[0].imshow(bragg_vector_map.data,vmax=50)
ax[1].imshow(bragg_vector_map_masked.data,vmax=50)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[0].set_title('bvm')
ax[1].set_title('bvm_masked')
plt.savefig(file_path_analysis+'bvm_raw_and_masked.png',dpi=300)




print('Vector map masked and saved')
plt.close('all') # Close all the open plots


## Calibrations- start with centering 

# Guess the center coordinate, specify the radial range for fitting peak pairs
#center_guess = (dataset.Qshape[0]//2, dataset.Qshape[1]//2 +2)
center_guess = (257//2, 257//2)
radial_range = (62,70)

# Show the ADF detector, overlaid over a new BVM
bragg_vector_map_masked = bragg_peaks_masked.get_bvm(mode='raw')

qxy_origins = bragg_peaks_masked.measure_origin(
    mode = 'no_beamstop',
    center_guess = center_guess,
    # radii = radial_range,
)

# Fit a plane to the origins
qx0_fit,qy0_fit,qx0_residuals,qy0_residuals = bragg_peaks_masked.fit_origin()

# apply the calibration
bragg_peaks_masked.calibration.set_origin((qx0_fit, qy0_fit))
bragg_peaks_masked.calibrate()

# Calculate BVM from centered data
bragg_vector_map_centered = bragg_peaks_masked.get_bvm()

print('Centered direct beam')
## Now for elipticity ##

q_range = (62,70)
# Fit the elliptical distortions
p_ellipse = py4DSTEM.process.calibration.fit_ellipse_1D(
    bragg_vector_map_centered,
    fitradii = q_range,
)

# # Apply the calibrations a
bragg_peaks_masked.calibration.set_p_ellipse(p_ellipse)
bragg_peaks_masked.calibrate()
print('Ellipse: ', p_ellipse)
print('Ellipticity Corrected')
## Can also do rotation calibration now but we havent bothered ##

# Save masked Bragg peaks
file_name_bragg_disks_masked_centered_circular = file_path_analysis + 'braggdisks_mcc.h5'
py4DSTEM.save(
    file_name_bragg_disks_masked_centered_circular,
    bragg_peaks_masked,
    mode='o',
)


    
    
f = open(file_path_analysis+'/p_ellipse.pickle', 'wb')   # Pickle file is newly created where foo1.py is
pickle.dump(p_ellipse, f)          # dump data to f
f.close()

print('Calibrations complete and bragg peaks saved')


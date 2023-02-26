import py4DSTEM
import pyxem as pxm
import matplotlib.pyplot as plt
import numpy as np
import os

inv_Ang_per_pixel = 0.01053034801747163


def pythag_distance(item):
        pythag_d= np.sqrt((item[0]**2)+(item[1]**2))
        return pythag_d
    
    
os.chdir(r'C:\Users\tas72\Documents\PhD\dg606\py4DSTEM_analysis\20221202_183724')


file_name_braggdisks_masked = r"C:\Users\tas72\Documents\PhD\dg606\py4DSTEM_analysis\20221202_183724\braggdisks_mcc.h5"
print(file_name_braggdisks_masked)
py4DSTEM.io.print_h5_tree(file_name_braggdisks_masked)


bragg_peaks = py4DSTEM.read(
    'braggdisks_masked.h5', 
    root = '/4DSTEM/braggvectors_copy'
)


bragg_peaks_calibrated = np.zeros((bragg_peaks.shape[0],bragg_peaks.shape[1]),dtype='object')
for i in range(bragg_peaks.shape[0]):
    print(i, end=' ')
    for j in range(bragg_peaks.shape[1]):
        brag_peaks_oi = bragg_peaks._v_uncal.get_pointlist(i,j).data
        calibrated_peaks = np.zeros_like(brag_peaks_oi) #currently peaks are in pixels and 0,0 is not at the center
        for k in range(len(brag_peaks_oi)):
            peak = brag_peaks_oi[k]
            x_coord = peak[0]
            y_coord = peak[1]
            x_coord = x_coord-128
            y_coord = y_coord-128
            x_coord = x_coord*inv_Ang_per_pixel
            y_coord = y_coord*inv_Ang_per_pixel
            calibrated_peaks[k][0] = x_coord
            calibrated_peaks[k][1] = y_coord
            calibrated_peaks[k][2] = peak[2]
            
        bragg_peaks_calibrated[i,j] = calibrated_peaks
    
print('Bragg Peaks Calibrated')
        
peaks_sorted = np.zeros_like(bragg_peaks_calibrated,dtype='object')
lengths_sorted = np.zeros_like(bragg_peaks_calibrated,dtype='object')
for i in range(np.shape(bragg_peaks_calibrated)[0]):
    print(i, end=' ')
    for j in range(np.shape(bragg_peaks_calibrated)[1]):
        nav_axis_oi = bragg_peaks_calibrated[i,j]
        lengths = np.zeros(len(nav_axis_oi))
        for k in range(len(lengths)):
            x,y = nav_axis_oi[k][0],nav_axis_oi[k][1]
            lengths[k] = pythag_distance((x,y))
        #np.argsort(lengths)
        sorted_data = np.zeros_like(nav_axis_oi)
        sorted_lengths = np.zeros_like(lengths)
        p=0
        for ind in np.argsort(lengths):
            sorted_data[p] = nav_axis_oi[ind]
            sorted_lengths[p] = lengths[ind]  
            p+=1
        peaks_sorted[i,j] = sorted_data
        lengths_sorted[i,j] = sorted_lengths
        
print('Bragg peaks sorted')
all_adj_matrix = np.zeros((np.shape(peaks_sorted)[0], np.shape(peaks_sorted)[1]),dtype = 'object')
for q in range(np.shape(peaks_sorted)[0]):
    print(q)
    for w in range(np.shape(peaks_sorted)[1]):
        print(w, end=' ' )
        
        nav_oi = lengths_sorted[q,w]
        groups = []
        for i in range(len(nav_oi)-1):    
            try:
                point = nav_oi[i]
                arg = np.argwhere(nav_oi>point+0.01)
                groups.append([np.amin(arg)])
            except:
                pass
        groups = np.unique(groups)
        groups = np.insert(groups,0,0)

        points_oi = peaks_sorted[q,w]
        lengths_oi = lengths_sorted[q,w]

        
        adj_matrix_points_list = []

        for i in range(len(groups)):
            if i == 0: # -0 is still 0
                continue

            group_start = groups[-i]    
            group_end = groups[(-i+1)] # the end group is where the group goes until    
            start_of_group_below = groups[-i-1]


            if group_end == 0: # put a catch for the end group
                group_end = len(lengths_sorted[q,w])


            lengths_oi = lengths_sorted[q,w][group_start:group_end]
            points_oi = peaks_sorted[q,w][group_start:group_end]




            points_oi_group_below = peaks_sorted[q,w][start_of_group_below:group_start]


            if i == 1: # If on first round define a random start
                poi = points_oi[0]
                loi = lengths_oi[0]



            complete_pythag_d = []

            for j in range(len(points_oi_group_below)):
                x_below = points_oi_group_below[j][0]
                y_below = points_oi_group_below[j][1]
                pd = pythag_distance((poi[0]-x_below, poi[1]-y_below))
                
                complete_pythag_d.append(pd)

            where_possible_points_for_next_round = np.where(np.round(complete_pythag_d,4) == np.round(np.amin(complete_pythag_d),4))




            if i == 1: # Dont have a previous point in the last round
                adj_matrix_points_list.append(poi)
                adj_matrix_points_list.append(points_oi_group_below[where_possible_points_for_next_round[0][0]])
                previous_poi = poi
                poi =  points_oi_group_below[where_possible_points_for_next_round[0][0]]

            else: # If at the first point shouldn't matter which 

                ## Want the point which is closest to the previous point
                distance_from_prior_point = []

                for j in range(len(where_possible_points_for_next_round[0])):
                    x_below = points_oi_group_below[where_possible_points_for_next_round[0][j]][0]
                    y_below = points_oi_group_below[where_possible_points_for_next_round[0][j]][1]
                    pd = pythag_distance((previous_poi[0]-x_below, previous_poi[1]-y_below))
                    distance_from_prior_point.append(pd)

                ## Make the previous_poi the current one from this round
                adj_matrix_points_list.append(points_oi_group_below[where_possible_points_for_next_round[0][0]])

                previous_poi = poi
                ## Make the new poi the one that has the minimum distance from the previous one
                poi = points_oi_group_below[where_possible_points_for_next_round[0][np.argmin(distance_from_prior_point)]]
                
                
            
            adj_matrix = np.zeros((len(adj_matrix_points_list),len(adj_matrix_points_list)))


            for i in range(np.shape(adj_matrix_points_list)[0]):
                current_poi = np.flip(adj_matrix_points_list)[i]
                other_points = [x for x in range(np.shape(adj_matrix_points_list)[0])]
                other_points.remove(i)

                for j in range(len(other_points)):
                    other_point = np.flip(adj_matrix_points_list)[other_points[j]]
                    pd = pythag_distance((current_poi[0]-other_point[0], current_poi[1]-other_point[1]))
                    adj_matrix[i,other_points[j]] = pd
        

                
                
                
            all_adj_matrix[q,w] = adj_matrix
                
                
np.save('complete_adj_matrix',all_adj_matrix)
                
print('Complete')

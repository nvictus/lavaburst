from collections import defaultdict
from scipy.signal import find_peaks_cwt
import numpy as np
from .utils import fill_tril_inplace


def call_boundary_peaks(Pb):
    n = len(Pb)
    regions = []

    peaks = find_peaks_cwt(
        Pb, np.array([0.5]), 
        wavelet=None, 
        max_distances=None, 
        gap_thresh=None, 
        min_length=None, 
        min_snr=1, 
        noise_perc=10)

    for i in peaks:
        top = Pb[i]
        x = [l for l in range(1, 5) if Pb[i-l] > 0.4*top]
        y = [r for r in range(1, 5) if Pb[i+r] > 0.4*top]
        start = (i - x[-1]) if len(x) else i
        end = (i + y[-1]) if len(y) else i
        regions.append([start-1, end+1, top])

    return map(np.array, zip(*regions))


def call_domain_peaks(Ps, thresh):
    from scipy.ndimage.filters import maximum_filter
    from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

    # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
    # answer by user: http://stackoverflow.com/users/50135/ivan
    def detect_peaks(image):
        """
        Takes an image and detect the peaks usingthe local maximum filter.
        Returns a boolean mask of the peaks (i.e. 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise)
        """

        # define an 8-connected neighborhood
        neighborhood = generate_binary_structure(2,2)

        #apply the local maximum filter; all pixel of maximal value 
        #in their neighborhood are set to 1
        local_max = maximum_filter(image, footprint=neighborhood) == image
        #local_max is a mask that contains the peaks we are 
        #looking for, but also the background.
        #In order to isolate the peaks we must remove the background from the mask.

        #we create the mask of the background
        background = (image==0)

        #a little technicality: we must erode the background in order to 
        #successfully subtract it form local_max, otherwise a line will 
        #appear along the background border (artifact of the local maximum filter)
        eroded_background = binary_erosion(
            background, structure=neighborhood, border_value=1)

        #we obtain the final mask, containing only peaks, 
        #by removing the background from the local_max mask
        detected_peaks = local_max - eroded_background

        return detected_peaks

    Ps_thresholded = np.array(Ps)
    Ps_thresholded[Ps_thresholded < thresh] = 0.0
    Ps_thresholded = fill_tril_inplace(
        Ps_thresholded, k=1, value=0.0)
    peak = detect_peaks(Ps_thresholded)
    
    x,y = np.where(peak)
    p = Ps_thresholded[x,y]

    return x, y, p


def consensus_domains(domains, weights):
    """
    Returns consensus list of nonoverlapping segments.
    Segments are 2-tuples given as half-open intervals [a,b).

    """
    occ = defaultdict(int)
    for d, w in zip(segments, weights):
        occ[d] += w

    # map each domain to its closest non-overlapping predecessor
    M = len(segments)
    prev = np.zeros(M, dtype=int)
    for i in range(M-1, -1, -1):
        d = segments[i]
        j = i - 1
        while j > -1:
            if segments[j][1] <= d[0]: 
                prev[i] = j
                break
            j -= 1

    # weighted interval scheduling dynamic program
    score = np.zeros(M, dtype=int)
    for i in range(1, M):
        d = segments[i]
        s_choose = score[prev[i]] + occ[d]
        s_ignore = score[i-1]
        score[i] = max(s_choose, s_ignore)

    consensus = []
    j = M - 1
    while j > 0:
        if score[j] != score[j-1]:
            consensus.append(segments[j])
            j = prev[j]
        else:
            j -= 1

    return consensus[::-1]

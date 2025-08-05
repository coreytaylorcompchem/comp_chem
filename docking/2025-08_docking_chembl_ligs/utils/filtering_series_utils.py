import numpy as np


def affinity_distribution(affinity_value):
    affinity_value = np.array(affinity_value, dtype=np.float32)
    if len(affinity_value) < 20:
        return False
    else:
        affinity_value = -np.sort(-affinity_value)
        affinity_value_fifth_max = affinity_value[2]
        affinity_value_fifth_min = affinity_value[-3]
        sub_range = affinity_value_fifth_max - affinity_value_fifth_min
        if sub_range > 2.5:
            if affinity_value_fifth_max > 7:
                return True
        return False


def nb_lig_with_mw_in_interval(mw_array):
    mw_array = np.array(mw_array, dtype=np.float32)
    array_cutoff = np.where(mw_array < 450, 1, 0)
    sum_nb_ligand_in_interval = np.sum(array_cutoff)
    return sum_nb_ligand_in_interval


def affinity_range_value(affinity_range):
    affinity_range = np.array(affinity_range, dtype=np.float32)
    range_value = np.max(affinity_range) - np.min(affinity_range)
    return range_value

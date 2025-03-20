import numpy as np
import pandas as pd
import copy
from scipy import interpolate


def wrapToPi(rad):
    rad_wrap = copy.copy(rad)
    q = (rad_wrap < -np.pi) | (np.pi < rad_wrap)
    rad_wrap[q] = ((rad_wrap[q] + np.pi) % (2 * np.pi)) - np.pi
    return rad_wrap

def log_interp1d(xx, yy, kind='linear'):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = interpolate.interp1d(logx, logy, kind=kind)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp


def segment_from_df(df, column='time', val=0.0):
    """ Pulls out segments from data frame based on where 'val' shows up in 'column'.
    """

    # Fins start of segments
    segment_start = np.where(df[column].values == val)[0].squeeze()  # where segments start
    n_segment = segment_start.shape[0]  # number of segments

    segment_list = []  # list of individual segments
    for n in range(n_segment):
        if n == (n_segment - 1):
            segment_end = df.shape[0]
        else:
            segment_end = segment_start[n + 1]

        segment = df.iloc[segment_start[n]:segment_end, :]
        segment_list.append(segment)

    return segment_list, n_segment


def collect_offset_rows(df, aug_column_names=None, keep_column_names=None, w=1, direction='backward'):
    """ Takes a pandas data frame with n rows, list of columns names, and a window size w.
        Then creates an augmented data frame that collects prior or future rows (in window)
        and stacks them as new columns. The augmented data frame will be size (n - w - 1) as the first/last
        w rows do not have enough data before/after them.

        Inputs
            df: pandas data frame
            aug_column_names: names of the columns to augment
            keep_column_names: names of the columns to keep, but not augment
            w: lookback window size (# of rows)
            direction: get the rows from behind ('backward') or front ('forward')

        Outputs
            df_aug: augmented pandas data frame.
                    new columns are named: old_name_0, old_name_1, ... , old_name_w-1
    """

    df = df.reset_index(drop=True)

    # Default for testing
    if df is None:
        df = np.atleast_2d(np.arange(0, 11, 1, dtype=np.double)).T
        df = np.matlib.repmat(df, 1, 4)
        df = pd.DataFrame(df, columns=['a', 'b', 'c', 'd'])
        aug_column_names = ['a', 'b']
    else:  # use the input  values
        # Default is all columns
        if aug_column_names is None:
            aug_column_names = df.columns

    # Make new column names & dictionary to store data
    new_column_names = {}
    df_aug_dict = {}
    for a in aug_column_names:
        new_column_names[a] = []
        df_aug_dict[a] = []

    for a in aug_column_names:  # each augmented column
        for k in range(w):  # each point in lookback window
            new_column_names[a].append(a + '_' + str(k))

    # Augment data
    n_row = df.shape[0]  # # of rows
    n_row_train = n_row - w + 1  # # of rows in augmented data
    for a in aug_column_names:  # each column to augment
        data = df.loc[:, [a]]  # data to augment
        data = np.asmatrix(data)  # as numpy matrix
        df_aug_dict[a] = np.nan * np.ones((n_row_train, len(new_column_names[a])))  # new augmented data matrix

        # Put augmented data in new column, for each column to augment
        for i in range(len(new_column_names[a])):  # each column to augment
            if direction == 'backward':
                # Start index, starts at the lookback window size & slides up by 1 for each point in window
                startI = w - 1 - i

                # End index, starts at end of the matrix &  & slides up by 1 for each point in window
                endI = n_row - i  # end index, starts at end of matrix &

            elif direction == 'forward':
                # Start index, starts at the beginning of matrix & slides up down by 1 for each point in window
                startI = i

                # End index, starts at end of the matrix minus the window size
                # & slides down by 1 for each point in window
                endI = n_row - w + 1 + i  # end index, starts at end of matrix &

            else:
                raise Exception("direction must be 'forward' or 'backward'")

            # Put augmented data in new column
            df_aug_dict[a][:, i] = np.squeeze(data[startI:endI, :])

        # Convert data to pandas data frame & set new column names
        df_aug_dict[a] = pd.DataFrame(df_aug_dict[a], columns=new_column_names[a])

    # Combine augmented column data
    df_aug = pd.concat(list(df_aug_dict.values()), axis=1)

    # Add non-augmented data, if specified
    if keep_column_names is not None:
        for c in keep_column_names:
            if direction == 'backward':
                startI = w - 1
                endI = n_row
            elif direction == 'forward':
                startI = 0
                endI = n_row - w
            else:
                raise Exception("direction must be 'forward' or 'backward'")

            keep = df.loc[startI:endI, [c]].reset_index(drop=True)
            df_aug = pd.concat([df_aug, keep], axis=1)

    return df_aug


def get_indices(fisher_data_structure, states_list=None, sensors_list=None, time_steps_list=None):
    """Get indices in data structure corresponding to states, sensors, & time-steps.
    """

    data = fisher_data_structure
    index_map = np.NaN * np.zeros((len(sensors_list), len(states_list), len(time_steps_list)))
    n_cond = len(data['states'])
    for j in range(n_cond):
        for n, states in enumerate(states_list):
            if states == data['states'][j]:
                for p, sensors in enumerate(sensors_list):
                    if sensors == data['sensors'][j]:
                        for k, time_steps in enumerate(time_steps_list):
                            if time_steps == data['time_steps'][j]:
                                index_map[p, n, k] = j

    # Check
    if np.sum(np.isnan(index_map)) > 0:
        # print('Input states, sensors, or time-steps that do not exist')
        raise Exception('Input states, sensors, or time-steps that do not exist')
    else:
        index_map = index_map.astype(int)

    n_sensors, n_states, n_time_steps = index_map.shape

    return index_map, n_sensors, n_states, n_time_steps

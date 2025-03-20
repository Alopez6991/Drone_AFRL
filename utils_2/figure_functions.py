import numpy as np
import numpy.matlib
import scipy
import pandas as pd
from fractions import Fraction
import matplotlib as mpl
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.transforms as transforms
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import figurefirst as fifi
import fly_plot_lib_plot as fpl
import util


def make_color_map(color_list=None, color_proportions=None, N=256):
    """ Make a colormap from a list of colors.
    """

    if color_list is None:
        color_list = ['white', 'deepskyblue', 'mediumblue', 'yellow', 'orange', 'red', 'darkred']

    if color_proportions is None:
        color_proportions = np.linspace(0.01, 1, len(color_list) - 1)
        v = np.hstack((np.array(0.0), color_proportions))
    elif color_proportions == 'even':
        color_proportions = np.linspace(0.0, 1, len(color_list))
        v = color_proportions

    l = list(zip(v, color_list))
    cmap = LinearSegmentedColormap.from_list('rg', l, N=N)

    return cmap


class LatexStates:
    """Holds LaTex format corresponding to set symbolic variables.
    """

    def __init__(self):
        self.dict = {'v_para': r'$v_{\parallel}$',
                     'v_perp': r'$v_{\perp}$',
                     'phi': r'$\phi$',
                     'phidot': r'$\dot{\phi}$',
                     'phi_dot': r'$\dot{\phi}$',
                     'phiddot': r'$\ddot{\phi}$',
                     'w': r'$w$',
                     'zeta': r'$\zeta$',
                     'w_dot': r'$\dot{w}$',
                     'zeta_dot': r'$\dot{\zeta}$',
                     'I': r'$I$',
                     'm': r'$m$',
                     'C_para': r'$C_{\parallel}$',
                     'C_perp': r'$C_{\perp}$',
                     'C_phi': r'$C_{\phi}$',
                     'km1': r'$k_{m_1}$',
                     'km2': r'$k_{m_2}$',
                     'km3': r'$k_{m_3}$',
                     'km4': r'$k_{m_4}$',
                     'd': r'$d$',
                     'psi': r'$\psi$',
                     'gamma': r'$\gamma$',
                     'alpha': r'$\alpha$',
                     'of': r'$\frac{g}{d}$',
                     'gdot': r'$\dot{g}$',
                     'v_para_dot': r'$\dot{v_{\parallel}}$',
                     'v_perp_dot': r'$\dot{v_{\perp}}$',
                     'v_para_dot_ratio': r'$\frac{\Delta v_{\parallel}}{v_{\parallel}}$',
                     'x':  r'$x$',
                     'y':  r'$y$',
                     'v_x': r'$v_{x}$',
                     'v_y': r'$v_{y}$',
                     'v_z': r'$v_{z}$',
                     'w_x': r'$w_{x}$',
                     'w_y': r'$w_{y}$',
                     'w_z': r'$w_{z}$',
                     'a_x': r'$a_{x}$',
                     'a_y': r'$a_{y}$',
                     'I_x': r'$I_{x}$',
                     'I_y': r'$I_{y}$',
                     'I_z': r'$I_{z}$',
                     'vx': r'$v_x$',
                     'vy': r'$v_y$',
                     'vz': r'$v_z$',
                     'wx': r'$w_x$',
                     'wy': r'$w_y$',
                     'wz': r'$w_z$',
                     'omega_x': r'$\omega_x$',
                     'omega_y': r'$\omega_y$',
                     'omega_z': r'$\omega_z$',
                     'ax': r'$ax$',
                     'ay': r'$ay$',
                     'thetadot': r'$\dot{\theta}$',
                     'theta_dot': r'$\dot{\theta}$',
                     'psidot': r'$\dot{\psi}$',
                     'psi_dot': r'$\dot{\psi}$',
                     'theta': r'$\theta$',
                     'Yaw': r'$\psi$',
                     'R': r'$\phi$',
                     'P': r'$\theta$',
                     'dYaw': r'$\dot{\psi}$',
                     'dP': r'$\dot{\theta}$',
                     'dR': r'$\dot{\phi}$',
                     'acc_x': r'$\dot{v}x$',
                     'acc_y': r'$\dot{v}y$',
                     'acc_z': r'$\dot{v}z$',
                     'Psi': r'$\Psi$',
                     'Ix': r'$I_x$',
                     'Iy': r'$I_y$',
                     'Iz': r'$I_z$',
                     'Jr': r'$J_r$',
                     'Dl': r'$D_l$',
                     'Dr': r'$D_r$',
                     }

    def convert_to_latex(self, list_of_strings, remove_dollar_signs=False):
        """ Loop through list of strings and if any match the dict, then swap in LaTex symbol.
        """

        if isinstance(list_of_strings, str):  # if single string is given instead of list
            list_of_strings = [list_of_strings]
            string_flag = True
        else:
            string_flag = False

        list_of_strings = list_of_strings.copy()
        for n, s in enumerate(list_of_strings):  # each string in list
            for k in self.dict.keys():  # check each key in Latex dict
                if s == k:  # string contains key
                    list_of_strings[n] = self.dict[k]  # replace string with LaTex
                    if remove_dollar_signs:
                        list_of_strings[n] = list_of_strings[n].replace('$', '')

        if string_flag:
            list_of_strings = list_of_strings[0]

        return list_of_strings


def plot_trajectory(xpos, ypos, phi, color, ax=None, size_radius=None, nskip=0,
                    colormap=None, colornorm=None, edgecolor='none', reverse=False):
    if color is None:
        color = phi

    color = np.array(color)

    # Set size radius
    xymean = np.mean(np.abs(np.hstack((xpos, ypos))))
    if size_radius is None:  # auto set
        xymean = 0.21 * xymean
        if xymean < 0.0001:
            sz = np.array(0.01)
        else:
            sz = np.hstack((xymean, 1))
        size_radius = sz[sz > 0][0]
    else:
        if isinstance(size_radius, list):  # scale defualt by scalar in list
            xymean = size_radius[0] * xymean
            sz = np.hstack((xymean, 1))
            size_radius = sz[sz > 0][0]
        else:  # use directly
            size_radius = size_radius

    if colornorm is None:
        colornorm = [np.min(color), np.max(color)]

    if reverse:
        xpos = np.flip(xpos, axis=0)
        ypos = np.flip(ypos, axis=0)
        phi = np.flip(phi, axis=0)
        color = np.flip(color, axis=0)

    if colormap is None:
        colormap = cm.get_cmap('bone_r')
        colormap = colormap(np.linspace(0.1, 1, 10000))
        colormap = ListedColormap(colormap)

    if ax is None:
        fig, ax = plt.subplots()

    fpl.colorline_with_heading(ax, np.flip(xpos), np.flip(ypos), np.flip(color, axis=0), np.flip(phi),
                               nskip=nskip,
                               size_radius=size_radius,
                               deg=False,
                               colormap=colormap,
                               center_point_size=0.0001,
                               colornorm=colornorm,
                               show_centers=False,
                               size_angle=20,
                               alpha=1,
                               edgecolor=edgecolor)

    ax.set_aspect('equal')
    xrange = xpos.max() - xpos.min()
    xrange = np.max([xrange, 0.02])
    yrange = ypos.max() - ypos.min()
    yrange = np.max([yrange, 0.02])

    if yrange < (size_radius / 2):
        yrange = 10

    if xrange < (size_radius / 2):
        xrange = 10

    ax.set_xlim(xpos.min() - 0.2 * xrange, xpos.max() + 0.2 * xrange)
    ax.set_ylim(ypos.min() - 0.2 * yrange, ypos.max() + 0.2 * yrange)

    # fifi.mpl_functions.adjust_spines(ax, [])


def pi_axis(ax, axis_name='y', tickpispace=0.5, lim=None, real_lim=None):
    """
    Format the specified axis with ticks labeled as multiples of π.

    Args:
        param ax: matplotlib axis object. The axis on which to apply the formatting.
        param axis_name: str, optional. The name of the axis to format ('x', 'y', or 'z'). Default is 'y'.
        param tickpispace: float, optional. The spacing between ticks as a multiple of π. Default is 0.5.
        param lim: tuple or None, optional. The limits for the axis. If None, defaults to (-π, π).
        param real_lim: tuple or None, optional. The real limits to set after formatting. If None, slightly extends beyond lim.
    """

    # Set axis limits
    default_lim = (-np.pi, np.pi)
    if lim is None:
        lim = default_lim

    set_lim = getattr(ax, f'set_{axis_name}lim')
    get_lim = getattr(ax, f'get_{axis_name}lim')
    set_ticks = getattr(ax, f'set_{axis_name}ticks')
    set_ticklabels = getattr(ax, f'set_{axis_name}ticklabels')

    set_lim(lim)
    lim = get_lim()

    # Generate tick positions and labels
    ticks = np.arange(lim[0], lim[1] + 0.01, tickpispace * np.pi)
    tickpi = np.round(ticks / np.pi, 3)
    y0 = abs(tickpi) < np.finfo(float).eps  # Identify zero entry

    tickslabels = ['$' + str(Fraction(val)) + r'\pi $' for val in tickpi]
    tickslabels = np.asarray(tickslabels, dtype=object)
    tickslabels[y0] = '0'  # Replace 0 entry with '0'

    set_ticks(ticks)
    set_ticklabels(tickslabels)

    # Adjust real limits if provided
    if real_lim is None:
        real_lim = (lim[0] - 0.4, lim[1] + 0.4)

    set_lim(real_lim)


def circplot(t, phi, jump=np.pi):
    """ Stitches t and phi to make unwrapped circular plot. """

    t = np.squeeze(t)
    phi = np.squeeze(phi)

    difference = np.abs(np.diff(phi, prepend=phi[0]))
    ind = np.squeeze(np.array(np.where(difference > jump)))

    phi_stiched = np.copy(phi)
    t_stiched = np.copy(t)
    for i in range(phi.size):
        if np.isin(i, ind):
            phi_stiched = np.concatenate((phi_stiched[0:i], [np.nan], phi_stiched[i + 1:None]))
            t_stiched = np.concatenate((t_stiched[0:i], [np.nan], t_stiched[i + 1:None]))

    return t_stiched, phi_stiched


def plot_heatmap_log_timeseries(data, log_ticks=None, data_labels=None,
                                fig=None, ax=None, cmap='inferno_r', y_label=None):
    """ Plot log-scale time-series as heatmap.
    """

    n_label = data.shape[1]

    # Set ticks
    if log_ticks is None:
        log_tick_low = int(np.floor(np.log10(np.min(data))))
        log_tick_high = int(np.ceil(np.log10(np.max(data))))
    else:
        log_tick_low = log_ticks[0]
        log_tick_high = log_ticks[1]

    log_ticks = np.logspace(log_tick_low, log_tick_high, log_tick_high - log_tick_low + 1)

    # Set color normalization
    cnorm = mpl.colors.LogNorm(10 ** log_tick_low, 10 ** log_tick_high)

    # Set labels
    if data_labels is None:
        data_labels = np.arange(0, n_label).tolist()
        data_labels = [str(x) for x in data_labels]

    # Make figure/axis
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(1, 1, figsize=(5 * 1, 4 * 1), dpi=150)
    elif fig is None:
        ax = plt.gca()

    # Plot heatmap
    ax.imshow(data, norm=cnorm, aspect=0.25, cmap=cmap, interpolation='none')

    # Set axis properties
    ax.grid(True, axis='x')
    ax.tick_params(axis='both', which='both', labelsize=6, top=False, labeltop=True, bottom=False, labelbottom=False,
                   color='gray')

    # Set x-ticks
    LatexConverter = LatexStates()
    data_labels_latex = LatexConverter.convert_to_latex(data_labels)
    ax.set_xticks(np.arange(0, len(data_labels)) - 0.5)
    ax.set_xticklabels(data_labels_latex)

    # Set labels
    ax.set_ylabel('time steps', fontsize=7, fontweight='bold')
    ax.set_xlabel('states', fontsize=7, fontweight='bold')
    ax.xaxis.set_label_position('top')

    # Set x-ticks
    xticks = ax.get_xticklabels()
    for tick in xticks:
        tick.set_ha('center')
        tick.set_va('center')
        tick.set_rotation(0)
        tick.set_transform(tick.get_transform() + transforms.ScaledTranslation(6 / 72, 0, ax.figure.dpi_scale_trans))

    # Colorbar
    if y_label is None:
        y_label = 'values'

    cax = ax.inset_axes((1.03, 0.0, 0.04, 1.0))
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap), cax=cax, ticks=log_ticks)
    cbar.set_label(y_label, rotation=270, fontsize=7, labelpad=8)
    cbar.ax.tick_params(labelsize=6)

    # Set the color of each spine individually
    for spine in ['bottom', 'top', 'left', 'right']:
        ax.spines[spine].set_color('gray')

    return cnorm, cmap, log_ticks


def plot_trajectory_error_variance(data_dict, states, sensors, time_steps,
                                   plot_vars=None, time_window=None,
                                   states_index=0, time_step_index=4,
                                   log_tick_low=-2, log_tick_high=6):
    """ Plot.
    """

    cmap = Colormaps(color_dict='purple_green_gray', power=3, flip_linear=False, flip_log=True)

    n_time_steps = len(time_steps)
    n_plot_var = len(plot_vars)

    log_ticks = np.logspace(log_tick_low, log_tick_high, log_tick_high - log_tick_low + 1)

    # Set colormaps
    cmap_time = plt.get_cmap('bone_r')
    cmap_time_rgb = cmap_time(np.linspace(0.1, 1, num=n_time_steps))
    cmap_time = ListedColormap(cmap_time_rgb)

    # Get index map
    index_map, n_sensors, n_states, n_time_steps = util.get_indices(data_dict,
                                                                    states_list=states,
                                                                    sensors_list=sensors,
                                                                    time_steps_list=time_steps)

    LatexConverter = LatexStates()

    # plot
    fig = plt.figure(figsize=((2.0 ** 3) * 2, 1.8 * n_sensors), dpi=300)
    subfigs = fig.subfigures(1, 3, wspace=0.07, width_ratios=[3.0, 3.0, 3.0], height_ratios=[1.0])

    ax_traj = subfigs[0].subplots(nrows=n_sensors, ncols=n_plot_var, sharex=True, sharey=True)
    ax_error_var = subfigs[1].subplots(nrows=n_sensors, ncols=n_plot_var, sharex=True, sharey=True)
    ax_time_step = subfigs[2].subplots(nrows=n_sensors, ncols=n_plot_var, sharex=True, sharey=True)


    cnorm = mpl.colors.LogNorm(10**log_tick_low, 10**log_tick_high)
    cnorm_time_step = mpl.cm.ScalarMappable(norm=plt.Normalize(vmin=time_steps[0], vmax=time_steps[-1]),
                                            cmap=cmap_time)

    data = data_dict.copy()
    for p in range(n_sensors):
        for k, v in enumerate(plot_vars):
            # Get sim data
            j = index_map[p, states_index, time_step_index]
            sim_data_traj = data['sim_data'][j]

            # Set time window
            t_sim = sim_data_traj['time'].values
            if time_window is None:
                time_window = (np.min(t_sim), np.max(t_sim))

            t_start = np.where((t_sim >= time_window[0]))[0][0]
            t_end = np.where((t_sim <= time_window[1]))[0][-1]
            sim_data_traj = sim_data_traj.iloc[t_start:t_end, :]

            # Get error covariance
            cvar = sim_data_traj['o_' + v].values

            # Plot
            heading = sim_data_traj['psi'].values
            x = sim_data_traj['x'].values
            y = sim_data_traj['y'].values

            x = x - np.mean(x)
            y = y - np.mean(y)

            # Plot
            plot_trajectory(x, y, heading,
                            color=cvar,
                            ax=ax_traj[p, k],
                            size_radius=0.08,
                            nskip=0,
                            colormap=cmap.log[v],
                            colornorm=cnorm)

            if k == 0:
                sensor_label = LatexConverter.convert_to_latex(data['sensors'][j])
                ax_traj[p, k].set_ylabel(', '.join(sensor_label), fontsize=8, loc='center')

            if p == 0:
                ax_traj[p, k].set_title(LatexConverter.convert_to_latex(v), fontsize=8, loc='center')

            # Colorline: log
            tsim = data['sim_data'][j]['time'].values
            ev = data['sim_data'][j]['o_' + v].values
            f = util.log_interp1d(tsim, ev, kind='slinear')
            tsim_interp = np.linspace(tsim[0], tsim[-1], 5000)
            ev_interp = f(tsim_interp)
            tsim_norm = np.linspace(tsim_interp[0], 1.0, tsim_interp.shape[0])

            colorline(tsim_norm, ev_interp, ev_interp,
                      ax=ax_error_var[p, k], cmap=cmap.log[v], norm=cnorm, linewidth=1.5)

            # ax_error_var[p, k].plot(tsim_interp, ev_interp, '.', markersize=1, linewidth=1, color='black')

            # Colorbar
            cax = ax_error_var[p, k].inset_axes([1.03, 0.0, 0.04, 1.0])
            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap.log[v]), cax=cax,
                                ticks=np.logspace(-2, 6, 9))
            cbar.set_label(LatexConverter.convert_to_latex(v) + ' error variance',
                           rotation=270, fontsize=7, labelpad=8)
            cbar.ax.tick_params(labelsize=6)

            # Plot time steps
            for t in range(n_time_steps):
                # Get error covariance
                j = index_map[p, states_index, t]
                sim_data = data['sim_data'][j]

                # Set time window
                t_sim = sim_data['time'].values
                if time_window is None:
                    time_window = (np.min(t_sim), np.max(t_sim))

                t_start = np.where((sim_data_traj.time.values >= time_window[0]))[0][0]
                t_end = np.where((sim_data_traj.time.values <= time_window[1]))[0][-1]
                sim_data_traj = sim_data_traj.iloc[t_start:t_end, :]

                # Plot
                # tsim_norm = np.linspace(t_sim[0], 0.96 + tsim[0], num=tsim.shape[0])
                ax_time_step[p, k].plot(t_sim, sim_data['o_' + v].values,
                                        linewidth=0.75, color=cmap_time_rgb[t, :])

            # Colorbar
            cax = ax_time_step[p, k].inset_axes([1.03, 0.0, 0.04, 1.0])
            cbar = fig.colorbar(cnorm_time_step, cax=cax, ticks=np.array(time_steps) + 1)
            cbar.set_label('window size', rotation=270, fontsize=7, labelpad=8)
            cbar.ax.tick_params(labelsize=6)

    for a in ax_traj[:, 0:2].flat:
        a.set_facecolor((1.0, 1.0, 1.0, 0.0))
        a.tick_params(axis='both', which='major', labelsize=6)
        # a.set_xlim(-0.1, 0.13)
        # a.set_ylim(-0.1, 0.1)
        fifi.mpl_functions.adjust_spines(a, [])

    for a in ax_error_var.flat:
        a.set_yscale('log')
        a.set_yticks(log_ticks)
        a.set_ylim(10**log_tick_low, 10**log_tick_high)
        # a.set_xlim(-0.02, 1.02)
        a.grid(linewidth=0.5)
        a.tick_params(axis='both', which='major', labelsize=6, left=False, labelleft=True)

    for a in ax_time_step.flat:
        a.set_yscale('log')
        a.set_yticks(log_ticks)
        a.set_ylim(10**log_tick_low, 10**log_tick_high)
        # a.set_xlim(-0.02, 1.02)
        a.grid(linewidth=0.5)
        a.tick_params(axis='both', which='major', labelsize=6, left=False, labelleft=True)

    for a in ax_error_var[-1, :].flat:
        a.set_xlabel('time (s)', fontsize=7)

    for a in ax_time_step[-1, :].flat:
        a.set_xlabel('time (s)', fontsize=7)

    subfigs[0].subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.2, hspace=-0.1)
    subfigs[1].subplots_adjust(left=-0.1, bottom=None, right=0.8, top=None, wspace=0.5, hspace=0.4)
    subfigs[2].subplots_adjust(left=-0.1, bottom=None, right=0.8, top=None, wspace=0.5, hspace=0.4)


def colorline(x, y, z, ax=None, cmap=plt.get_cmap('copper'), norm=None, linewidth=1.5, alpha=1.0):
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    # Set normalization
    if norm is None:
        norm = plt.Normalize(np.min(z), np.max(z))

    print(norm)

    # Make segments
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha,
                              path_effects=[path_effects.Stroke(capstyle="round")])

    # Plot
    if ax is None:
        ax = plt.gca()

    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def log_colormap(cmap, cmap_size=1e5, log_size=1e4, power=2, flip=False, plot=False):
    """Convert a colormap to log scale.
    """

    cmap_size = int(cmap_size)
    log_size = int(log_size)

    # Get the color map in RGB format
    cmap_rgb = cmap(np.linspace(0.0, 1, cmap_size))
    n_grid = cmap_rgb.shape[0]

    # Logarithmically spaced indices in color map
    cmap_index = np.linspace(0, 1, log_size)
    # log_space = np.logspace(0, np.log10(n_grid), num=log_size) - 1
    # log_space = np.round(log_space).astype(int)
    # log_space = 1**cmap_index
    # log_space = np.exp(cmap_index)
    log_space = cmap_index ** power
    log_space = (n_grid - 1) * (log_space / np.max(log_space))
    log_space = log_space.astype(int)

    # Plot log transform
    if plot:
        plt.plot(cmap_index, n_grid * log_space)

    # Make new colormap
    cmap_log = np.zeros((log_size, 4))
    for n, k in enumerate(log_space):
        cmap_log[n, :] = cmap_rgb[k, :]

    if flip:
        cmap_log = np.flipud(cmap_log)

    cmap_log = ListedColormap(cmap_log)

    return cmap_log


class LatexStates:
    """Holds LaTex format corresponding to set symbolic variables.
    """

    def __init__(self):
        self.dict = {'v_para': r'$v_{\parallel}$',
                     'v_perp': r'$v_{\perp}$',
                     'phi': r'$\phi$',
                     'phidot': r'$\dot{\phi}$',
                     'phi_dot': r'$\dot{\phi}$',
                     'phiddot': r'$\ddot{\phi}$',
                     'w': r'$w$',
                     'zeta': r'$\zeta$',
                     'w_dot': r'$\dot{w}$',
                     'zeta_dot': r'$\dot{\zeta}$',
                     'I': r'$I$',
                     'm': r'$m$',
                     'C_para': r'$C_{\parallel}$',
                     'C_perp': r'$C_{\perp}$',
                     'C_phi': r'$C_{\phi}$',
                     'km1': r'$k_{m_1}$',
                     'km2': r'$k_{m_2}$',
                     'km3': r'$k_{m_3}$',
                     'km4': r'$k_{m_4}$',
                     'd': r'$d$',
                     'psi': r'$\psi$',
                     'gamma': r'$\gamma$',
                     'alpha': r'$\alpha$',
                     'beta': r'$\beta$',
                     'of': r'$\frac{g}{d}$',
                     'gdot': r'$\dot{g}$',
                     'v_para_dot': r'$\dot{v_{\parallel}}$',
                     'v_perp_dot': r'$\dot{v_{\perp}}$',
                     'v_para_dot_ratio': r'$\frac{\Delta v_{\parallel}}{v_{\parallel}}$',
                     'x':  r'$x$',
                     'y':  r'$y$',
                     'v_x': r'$v_{x}$',
                     'v_y': r'$v_{y}$',
                     'v_z': r'$v_{z}$',
                     'w_x': r'$w_{x}$',
                     'w_y': r'$w_{y}$',
                     'w_z': r'$w_{z}$',
                     'a_x': r'$a_{x}$',
                     'a_y': r'$a_{y}$',
                     'vx': r'$v_x$',
                     'vy': r'$v_y$',
                     'vz': r'$v_z$',
                     'wx': r'$w_x$',
                     'wy': r'$w_y$',
                     'wz': r'$w_z$',
                     'omega_x': r'$\omega_x$',
                     'omega_y': r'$\omega_y$',
                     'omega_z': r'$\omega_z$',
                     'ax': r'$ax$',
                     'ay': r'$ay$',
                     'thetadot': r'$\dot{\theta}$',
                     'theta_dot': r'$\dot{\theta}$',
                     'psidot': r'$\dot{\psi}$',
                     'psi_dot': r'$\dot{\psi}$',
                     'theta': r'$\theta$',
                     'Yaw': r'$\psi$',
                     'R': r'$\phi$',
                     'P': r'$\theta$',
                     'dYaw': r'$\dot{\psi}$',
                     'dP': r'$\dot{\theta}$',
                     'dR': r'$\dot{\phi}$',
                     'acc_x': r'$\dot{v}x$',
                     'acc_y': r'$\dot{v}y$',
                     'acc_z': r'$\dot{v}z$',
                     'Psi': r'$\Psi$',
                     'Ix': r'$I_x$',
                     'Iy': r'$I_y$',
                     'Iz': r'$I_z$',
                     'Jr': r'$J_r$',
                     'Dl': r'$D_l$',
                     'Dr': r'$D_r$',
                     }

    def convert_to_latex(self, list_of_strings, remove_dollar_signs=False):
        """ Loop through list of strings and if any match the dict, then swap in LaTex symbol.
        """

        if isinstance(list_of_strings, str):  # if single string is given instead of list
            list_of_strings = [list_of_strings]
            string_flag = True
        else:
            string_flag = False

        list_of_strings = list_of_strings.copy()
        for n, s in enumerate(list_of_strings):  # each string in list
            for k in self.dict.keys():  # check each key in Latex dict
                if s == k:  # string contains key
                    list_of_strings[n] = self.dict[k]  # replace string with LaTex
                    if remove_dollar_signs:
                        list_of_strings[n] = list_of_strings[n].replace('$', '')

        if string_flag:
            list_of_strings = list_of_strings[0]

        return list_of_strings


class Colormaps:
    """ Creates colormaps for error covariance data.
    """

    def __init__(self, color_dict=None, power=5, flip_linear=False, flip_log=True):
        """ Run.
        """

        # Set colormaps
        self.color_list = None
        if color_dict is None:
            # zeta_color = np.array([70, 220, 40]) / 255
            # w_color = np.array([0, 110, 90]) / 255

            color_zeta = np.array([100, 250, 87]) / 255
            color_w = np.array([18, 139, 110]) / 255

            self.color_dict = {'zeta': ['honeydew', 'greenyellow', 'lawngreen', color_zeta],
                               'w': ['honeydew', 'mediumseagreen', 'seagreen', color_w],
                               'theta': ['whitesmoke', np.array([100, 100, 100]) / 255]}

        elif color_dict == 'purple_green_gray':
            stack = 0

            colors = [np.array([150.0, 150.0, 150.0]) / 255.0,
                      np.array([180.0, 180.0, 180.0]) / 255.0,
                      np.array([195.0, 117.0, 207.0]) / 255.0,
                      # np.array([184.0, 115.0, 255.0]) / 255.0,
                      # np.array([180.0, 109.0, 250.0]) / 255.0,
                      np.array([201.0, 35.0, 211.0]) / 255.0,
                      # np.array([168.0, 35.0, 211.0]) / 255.0,
                      np.array([142.0, 35.0, 211.0]) / 255.0,
                      np.array([136.0, 31.0, 240.0]) / 255.0,
                      # np.array([103.0, 13.0, 191.0]) / 255.0,
                      np.array([85.0, 4.0, 170.0]) / 255.0,
                      np.array([55.0, 2.0, 108.0]) / 255.0
                      ]

            cmap_zeta = LinearSegmentedColormap.from_list('purple_gray', colors, N=256)
            cmap_zeta = cmap_zeta(np.linspace(0.0, 1.0, num=1000)).tolist()

            colors = [np.array([150.0, 150.0, 150.0]) / 255.0,
                      np.array([180.0, 180.0, 180.0]) / 255.0,
                      np.array([78.0, 187.0, 228.0]) / 255.0,
                      # np.array([186.0, 197.0, 213.0]) / 255.0,
                      np.array([45.0, 213.0, 173.0]) / 255.0,
                      np.array([12.0, 175.0, 16.0]) / 255.0,
                      np.array([13.0, 144.0, 16.0]) / 255.0,
                      np.array([2.0, 97.0, 4.0]) / 255.0
                      ]

            cmap_w = LinearSegmentedColormap.from_list('green_gray', colors, N=256)
            cmap_w = cmap_w(np.linspace(0.0, 1.0, num=1000)).tolist()
            # cmap_w = np.vstack((cmap_w, np.matlib.repmat(cmap_w[-1], stack, 1))).tolist()

            cmap_v_para = mpl.colormaps['Reds'](np.linspace(0.0, 1.0, num=1000))
            cmap_v_para = np.vstack((cmap_v_para, np.matlib.repmat(cmap_v_para[-1], stack, 1))).tolist()

            cmap_d = mpl.colormaps['Blues'](np.linspace(0.0, 1.0, num=1000))
            cmap_d = np.vstack((cmap_d, np.matlib.repmat(cmap_d[-1], stack, 1))).tolist()

            self.color_dict = {'zeta': cmap_zeta, 'w': cmap_w, 'v_para': cmap_v_para, 'd': cmap_d}

            self.color_list = {'zeta': np.array([68.0, 21.0, 185.0]) / 255,
                               'w': np.array([22.0, 125.0, 8.0]) / 255,
                               'v_para': 'red',
                               'd': 'blue',
                               }
        else:
            self.color_dict = color_dict

        # Color list
        if self.color_list is None:
            self.color_list = {}
            for k in self.color_dict:
                self.color_list[k] = self.color_dict[k][-1]

        # Color
        self.linear = {}
        self.log = {}

        for k in self.color_dict.keys():
            if isinstance(self.color_dict[k], list):
                self.linear[k] = make_color_map(color_list=self.color_dict[k],
                                                color_proportions='even', N=256)
            else:
                self.linear[k] = self.color_dict[k]

            cmap = self.linear[k](np.linspace(0.1, 1.0, 1000))
            cmap = ListedColormap(cmap)
            self.log[k] = log_colormap(cmap, power=power, flip=flip_log)

            if flip_linear:
                self.color_dict[k].reverse()
                self.linear[k] = make_color_map(color_list=self.color_dict[k],
                                                color_proportions='even', N=256)
import matplotlib.pyplot as plt
import numpy as np
import fly_plot_lib.plot as fpl
import figurefirst as fifi
import pynumdiff
import pynumdiff.optimize
import pandas
import scipy.integrate
import copy
import math

def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))

def mean_angle(angle):
    return np.arctan2(np.nanmean(np.sin(angle)), np.nanmean(np.cos(angle)))
    
def std_angle(angle):
    return np.sqrt( np.sum(wrap_angle(angle - mean_angle(angle))**2) / len(angle) )

def unwrap_angle(z, correction_window_for_2pi=100, n_range=2, plot=False):
    if 0: # option one
        zs = []
        for n in range(-1*n_range, n_range):
            zs.append(z+n*np.pi*2)
        zs = np.vstack(zs)

        smooth_zs = np.array(z[0:2])

        for i in range(2, len(z)):
            first_ix = np.max([0, i-correction_window_for_2pi])
            last_ix = i
            error = np.abs(zs[:,i] - np.mean(smooth_zs[first_ix:last_ix])) 
            smooth_zs = np.hstack(( smooth_zs, [zs[:,i][np.argmin(error)]] ))

        if plot:
            for r in range(zs.shape[0]):
                plt.plot(zs[r,:], '.', markersize=1)
            plt.plot(smooth_zs, '.', color='black', markersize=1)
        
    else: # option two, automatically scales n_range to most recent value, and maybe faster
        smooth_zs = np.array(z[0:2])
        for i in range(2, len(z)):
            first_ix = np.max([0, i-correction_window_for_2pi])
            last_ix = i
            
            nbase = np.round( (smooth_zs[-1] - z[i])/(2*np.pi) )
            
            candidates = []
            for n in range(-1*n_range, n_range):
                candidates.append(n*2*np.pi+nbase*2*np.pi+z[i])
            error = np.abs(candidates - np.mean(smooth_zs[first_ix:last_ix])) 
            smooth_zs = np.hstack(( smooth_zs, [candidates[np.argmin(error)]] ))
        if plot:
            plt.plot(smooth_zs, '.', color='black', markersize=1)
    return smooth_zs

def plot_wind_quivers(df, ax=None, res=3, wind_quiver_index=0, headwidth=7, alpha=0.5):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    i = int(res/2.)
    
    xrang = df.xpos.max() - df.xpos.min()
    yrang = df.ypos.max() - df.ypos.min()

    x = np.arange(df.xpos.min()-0.2*xrang, df.xpos.max()+0.2*xrang, .1)
    y = np.arange(df.ypos.min()-0.2*yrang, df.ypos.max()+0.2*yrang, .1)
    X, Y = np.meshgrid(x, y)
    
    wind_x = df.w.values[wind_quiver_index]*np.cos(df.zeta.values[wind_quiver_index])
    wind_y = df.w.values[wind_quiver_index]*np.sin(df.zeta.values[wind_quiver_index])
    
    V = wind_x.mean()*np.ones_like(X[i:-1:res, i:-1:res])
    U = wind_y.mean()*np.ones_like(Y[i:-1:res, i:-1:res])
    
    q = ax.quiver(X[i:-1:res, i:-1:res], Y[i:-1:res, i:-1:res], V, U, alpha=alpha, headwidth=headwidth)

def plot_trajec(df, ax=None, size_radius=5, nskip = 190, 
                show_wind_quivers=True, wind_quiver_index=0, wind_quiver_res=3, wind_quiver_headwidth=7, wind_quiver_alpha=0.5, colormap='bone_r'):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if show_wind_quivers:
        plot_wind_quivers(df, ax, wind_quiver_index=wind_quiver_index, res=wind_quiver_res, headwidth=wind_quiver_headwidth, alpha=wind_quiver_alpha)
    
    fpl.colorline_with_heading(ax, df['xpos'].values, df['ypos'].values, df['t'].values, df['phi'].values, 
                                    nskip=nskip, size_radius=size_radius, deg=False, colormap=colormap, center_point_size=0.0001,
                                    colornorm=[0.05*df['t'].values[-1],df['t'].values[-1]], show_centers=False)

    ax.set_aspect('equal')
    xrang = df.xpos.max() - df.xpos.min()
    xrang = np.max([xrang, 0.1])
    yrang = df.ypos.max() - df.ypos.min()
    yrang = np.max([yrang, 0.1])
    ax.set_xlim(df.xpos.min()-0.1*xrang, df.xpos.max()+0.1*xrang)
    ax.set_ylim(df.ypos.min()-0.1*yrang, df.ypos.max()+0.1*yrang)

    fifi.mpl_functions.adjust_spines(ax, [])

def get_sensor_measurements_derivatives_controls(df_state, 
                                                 angular_noise_std=0,
                                                 of_noise_std=0,
                                                 air_noise_std=0,
                                                 sensor_group='polar', # or cartesian,
                                                 derivative_method='total_variation_regularization.position',
                                                 return_smoothed=False,
                                                 params='turning', # 'None': auto, 'turning': use twice turning freq., or hard code
                                                 cutoff_freq=0.1, # for auto parameters
                                                 correction_window_for_2pi=100,
                                                 phi_alignment='align_psi',
                                                 butterworth_freq_param_adjustment=1
                                                ):
    dt = np.mean(np.diff(df_state.t))
    
    def constrain_params(params):
                if params[1] < 1e-4:
                    params[1] = 1e-4
                if params[1] > 0.99:
                    params[1] = 0.99
                return params
    
    if sensor_group == 'cartesian':
        phi_noise = np.random.normal(0, angular_noise_std, len(df_state.sensor_psi))
        air_para_noise = np.random.normal(0, air_noise_std, len(df_state.sensor_psi))
        air_perp_noise = np.random.normal(0, air_noise_std, len(df_state.sensor_psi))
        of_para_noise = np.random.normal(0, of_noise_std, len(df_state.sensor_psi))
        of_perp_noise = np.random.normal(0, of_noise_std, len(df_state.sensor_psi))
        
        df_sensor = pandas.DataFrame({'sensor_phi': df_state.sensor_phi+phi_noise, 
                                      'sensor_air_para': df_state.air_para+air_para_noise, 
                                      'sensor_air_perp': df_state.air_perp+air_perp_noise, 
                                      'sensor_of_para': df_state.sensor_of_para+of_para_noise, 
                                      'sensor_of_perp': df_state.sensor_of_perp+of_perp_noise, 
                                      't': df_state.t})
        try:
            df_sensor['u_para'] = df_state.u_para
            df_sensor['u_perp'] = df_state.u_perp
            df_sensor['u_phi'] = df_state.u_phi
        except:
            pass
        
    elif sensor_group == 'polar':
        psi_noise = np.random.normal(0, angular_noise_std, len(df_state.sensor_psi))
        phi_noise = np.random.normal(0, angular_noise_std, len(df_state.sensor_psi))
        gamma_noise = np.random.normal(0, angular_noise_std, len(df_state.sensor_psi))
        
        if params is None:
            print('Automatically determine params based on pynumdiff.optimize')
            print('automatically determining differentiation parameters based on first 1000 indices')
            ixend = np.max([len(df_state.sensor_gamma.values)-1, 1000])
            tvgamma = np.exp(-1.6*np.log(cutoff_freq)-0.71*np.log(dt)-5.1)
            family, method = derivative_method.split('.')

            ## gamma
            params_gamma, res = pynumdiff.optimize.__dict__[family].__dict__[method](df_state.sensor_gamma.values[0:ixend], dt, tvgamma=tvgamma)
            params_gamma[1] *= butterworth_freq_param_adjustment
            params_gamma = constrain_params(params_gamma)
            print('optimal params gamma: ', params_gamma)
            sensor_gamma_smooth, sensor_gamma_dot = diff_angle(df_state.sensor_gamma.values+gamma_noise, dt, params_gamma, derivative_method=derivative_method, correction_window_for_2pi=correction_window_for_2pi)
            ##

            ## psi
            params_psi, res = pynumdiff.optimize.__dict__[family].__dict__[method](df_state.sensor_psi.values[0:ixend], dt, None, tvgamma=tvgamma)
            params_psi[1] *= butterworth_freq_param_adjustment
            params_psi = constrain_params(params_psi)
            print('optimal params psi: ', params_psi)
            sensor_psi_smooth, sensor_psi_dot = diff_angle(df_state.sensor_psi.values+psi_noise, dt, params_psi, derivative_method=derivative_method, correction_window_for_2pi=correction_window_for_2pi)
            ##

            ## phi
            params_phi, res = pynumdiff.optimize.__dict__[family].__dict__[method](df_state.sensor_phi.values[0:ixend], dt, None, tvgamma=tvgamma)
            params_phi[1] *= butterworth_freq_param_adjustment
            params_phi = constrain_params(params_phi)
            print('optimal params phi: ', params_phi)
            sensor_phi_smooth, sensor_phi_dot = diff_angle(df_state.sensor_phi.values+phi_noise, dt, params_phi, derivative_method=derivative_method, correction_window_for_2pi=correction_window_for_2pi)
            ## 
            
        elif params == 'turning':
            print('Automatically determine params based on 1x turning frequency')
            nyquist_freq = 0.5*1/dt
            butter_freq = cutoff_freq / nyquist_freq
            params = [4, butter_freq]
            params[1] *= butterworth_freq_param_adjustment
            
            params_gamma = copy.copy(params)
            params_psi = copy.copy(params)
            params_phi = copy.copy(params)
    
            #if phi_alignment == 'align_psi':
            #    params_psi[1] /= 10
            #if phi_alignment == 'align_gamma':
            #    params_gamma[1] /= 10
                
            params_gamma = constrain_params(params_gamma)
            params_psi = constrain_params(params_psi)
            params_phi = constrain_params(params_phi)
            
            print('Smoothing parameters')
            print('Turning freq: ' + str(cutoff_freq))
            print('Butter freq: ' + str(butter_freq))
            print('Params gamma, psi, phi: ')
            print(params_gamma)
            print(params_psi)
            print(params_phi)
            
            sensor_gamma_smooth, sensor_gamma_dot = diff_angle(df_state.sensor_gamma.values+gamma_noise, dt, params_gamma, derivative_method=derivative_method, correction_window_for_2pi=correction_window_for_2pi)
            sensor_psi_smooth, sensor_psi_dot = diff_angle(df_state.sensor_psi.values+psi_noise, dt, params_psi, derivative_method=derivative_method, correction_window_for_2pi=correction_window_for_2pi)
            sensor_phi_smooth, sensor_phi_dot = diff_angle(df_state.sensor_phi.values+phi_noise, dt, params_phi, derivative_method=derivative_method, correction_window_for_2pi=correction_window_for_2pi)
    
    
        if return_smoothed:
            df_sensor = pandas.DataFrame({'sensor_gamma': sensor_gamma_smooth, #wrap_angle(sensor_gamma_smooth),
                                          'sensor_gamma_dot': sensor_gamma_dot,
                                          'sensor_psi': sensor_psi_smooth, #wrap_angle(sensor_psi_smooth),
                                          'sensor_psi_dot': sensor_psi_dot,
                                          'sensor_phi': sensor_phi_smooth, #wrap_angle(sensor_phi_smooth),
                                          'sensor_phi_dot': sensor_phi_dot,
                                          't': df_state.t})
            try:
                df_sensor['u_para'] = df_state.u_para
                df_sensor['u_perp'] = df_state.u_perp
                df_sensor['u_phi'] = df_state.u_phi
            except:
                pass
            
        else:
            df_sensor = pandas.DataFrame({'sensor_gamma': df_state.sensor_gamma+gamma_noise, #wrap_angle(sensor_gamma_smooth),
                                          'sensor_gamma_dot': sensor_gamma_dot,
                                          'sensor_psi': df_state.sensor_psi+psi_noise, #wrap_angle(sensor_psi_smooth),
                                          'sensor_psi_dot': sensor_psi_dot,
                                          'sensor_phi': df_state.sensor_phi+phi_noise, #wrap_angle(sensor_phi_smooth),
                                          'sensor_phi_dot': sensor_phi_dot,
                                          'sensor_gamma_smooth': sensor_gamma_smooth,
                                          'sensor_psi_smooth': sensor_psi_smooth,
                                          'sensor_phi_smooth': sensor_phi_smooth,
                                          
                                          't': df_state.t})
            try:
                df_sensor['u_para'] = df_state.u_para
                df_sensor['u_perp'] = df_state.u_perp
                df_sensor['u_phi'] = df_state.u_phi
            except:
                pass

    parameters = pandas.DataFrame({'params_gamma': params_gamma,
                                   'params_psi': params_psi,
                                   'params_phi': params_phi})
                                   
    return df_sensor, parameters

def get_indices_for_outliers_angle(chunk, ix0=0, outlier_max_std=1.5):
    diffs = wrap_angle(chunk - mean_angle(chunk))
    ix = np.where( np.abs(diffs) > outlier_max_std*std_angle(chunk))[0]
    return ix + ix0

def interpolate_nans(y):
    if type(y) is list:
        y = np.array(y)
    nans = np.isnan(y)
    x = lambda z: z.nonzero()[0]
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    return y

def interpolate_outliers_angle(data, window_size=500, stride=250, outlier_max_std=1.5):
    data_cleaned = copy.copy(data)
    for ix0 in range(0, len(data)-window_size, stride):
        ix = get_indices_for_outliers_angle(data[ix0:ix0+window_size], ix0=ix0, outlier_max_std=outlier_max_std)
        data_cleaned[ix] = np.nan
    data_cleaned = interpolate_nans(data_cleaned)
    return data_cleaned

def diff_angle(angles, dt, params, 
               derivative_method='smooth_finite_difference.butterdiff', 
               outlier_max_std=1.5,
               outlier_window_size=500,
               outlier_stride=250,
               correction_window_for_2pi=100):
    '''
    Take a filtered derivative of an angle
    '''
    
    family, method = derivative_method.split('.')

    '''
    angles = interpolate_outliers_angle(angles, outlier_window_size, outlier_stride, outlier_max_std)
    diff_angles = np.diff(angles)
    diff_angles = np.hstack((0, diff_angles, 0))
    wrapped_diff_angle = wrap_angle(diff_angles)
    unwrapped_angle = scipy.integrate.cumtrapz(wrapped_diff_angle)
    
    corrected_unwrapped_angle = [unwrapped_angle[0], unwrapped_angle[1]]
    for i in range(2, len(unwrapped_angle)):
        first_ix = np.max([0, i-correction_window_for_2pi])
        last_ix = i
        error = (unwrapped_angle[i] - mean_angle(corrected_unwrapped_angle[first_ix:last_ix])) / (2*np.pi)
        npi = np.round(error)
        corrected_unwrapped_angle.append(unwrapped_angle[i] - npi*2*np.pi)
    
    offset = mean_angle(angles) - mean_angle(unwrapped_angle)
    unwrapped_angle += offset
    '''
    
    unwrapped_angle = unwrap_angle(angles, correction_window_for_2pi=correction_window_for_2pi, n_range=5)

    if family == 'total_variation_regularization' and method == 'position':
        angles_smooth, angles_dot = diff_tvrp(unwrapped_angle, dt, params)
        return wrap_angle(angles_smooth), angles_dot
    else:
        angles_smooth, angles_dot = pynumdiff.__dict__[family].__dict__[method](unwrapped_angle, dt, params, {})
        return wrap_angle(angles_smooth), angles_dot

def correct_wind_direction_for_trisonica(trisonica_zeta, declination=13*np.pi/180):
    # correct for declination (reno is 13 deg)
    corrected_zeta = trisonica_zeta - declination 
    
    # flip sign, because positive angles should be counter clockwise
    corrected_zeta *= -1
    
    # add pi/2, because north is pi/2, not zero
    corrected_zeta += np.pi/2.
    
    # add pi because we want wind described in terms of where it is blowing to, not from
    corrected_zeta += np.pi
    
    return wrap_angle(corrected_zeta)

def get_lat_lon_scale(origin, destination):
    """
    Calculate the Haversine distance.

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    >>> origin = (48.1372, 11.5756)  # Munich
    >>> destination = (52.5186, 13.4083)  # Berlin
    >>> round(distance(origin, destination), 1)
    504.2
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d*1000 # in meters
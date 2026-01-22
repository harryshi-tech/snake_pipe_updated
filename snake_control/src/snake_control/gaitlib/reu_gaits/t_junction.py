'''
NOTES : 
Have two amplitude commands - One before the bend and one after the bend
'''

from copy import deepcopy
import numpy as np

def t_junction(self, t=0, params=None, pole_params=None, compute=True):
    
    params = {} if params is None else params
    pole_params = {} if pole_params is None else pole_params

    self.current_gait = "t_junction"

    # Update the current parameters if params is not empty
    defaults = self.default_gait_params.get(self.current_gait) or {}
    self.current_gait_params = self.update_params(defaults, params)

    A_transition = pole_params.get("A_transition", 0.35)
    A_max = pole_params.get("A_max", 1.25)
    dWs_dAodd = pole_params.get("dWs_dAodd", 2.5 / 0.75)

    # TODO: the section below should be updated to reuse code from _sidewinding.
    # Update the gait parameters and then provie that to _sidewinding.

    n_samples = 1 if (isinstance(t, float) or isinstance(t, int)) else len(t)

    # Extract individual gait parameters from dictionary
    beta_even = self.current_gait_params['beta_even']
    beta_odd = self.current_gait_params['beta_odd']
    A_even = self.current_gait_params['A_even']
    A_odd = self.current_gait_params['A_odd']
    wS_even = self.current_gait_params['wS_even']
    wS_odd = self.current_gait_params['wS_odd']
    wT_even = self.current_gait_params['wT_even']
    wT_odd = self.current_gait_params['wT_odd']
    delta = self.current_gait_params['delta']
    wt_direction = self.current_gait_params['wt_direction']
    tightness = self.current_gait_params['tightness']
    pole_direction = self.current_gait_params['pole_direction']


    """ We start with spatial frequency being zero. Once the amplitude reaches
        amplitude_transition, we then set spatial frequency according to following line:
             ^
             |                     * * * * * * * * wS_odd
             |                   *
             |     dWs_dAodd   *
             |       --------*
         wS  |       |     *
             |       |   *
             |       | *
             |       *
    wS_min-->|     *
             |     *
             * * * *------------------------------>
                   ^           A_odd
                   |
                A_transition

    The "tightness" of the helix is therefore determined solely by the amplitude. It is
    also possible to adjust wS and the amplitude directly by deriving the Jacobian relating
    the radius of the helix and wS/amplitude. We leave this for future work.
    """

    """All this has to be handled one level higher in order to allow for an admittance controller on wS."""

    wS_max = wS_even # Assume wS_even to be wS_max from yaml.
    A_min = A_even
    # Update spatial frequency using commanded tightness.
    if tightness < A_transition:
        wS_odd = 0
    else:
        wS_odd = min(wS_max, (tightness - A_transition) * dWs_dAodd)

    wS_odd *= -pole_direction

    # Update amplitude using commanded tightness.
    if tightness < A_min:
        A_odd = A_min
    else:
        A_odd = min(tightness, A_max)

    A_odd *= -pole_direction

    wS_even = wS_odd
    A_even = A_odd
    target_angles = np.zeros(self.num_modules)
    target_angles_spiraling = np.zeros(self.num_modules)
    # unwrap = params['unwrap']
    wS_1 = wS_even
    wS_2 = wS_even
    A_1_multiplier = float(params.get('A_1_multiplier', 1.0))
    A_2_multiplier = float(params.get('A_2_multiplier', 1.0))
    mu = float(params.get('mu', (self.num_modules - 1) / 2.0))
    phi_0 = float(params.get('phi_0', 0.0))
    s_0 = float(params.get('s_0', 0.4))
    speed_multiplier = float(params.get('speed_multiplier', 1.0))
    debug_print = bool(params.get('debug_print', False))

    A_1 = A_even * A_1_multiplier
    A_2 = A_even * A_2_multiplier

    """vvvv BLock of code for spiraling gait vvvv"""
    if self.snake_type == "SEA":
        module_length = 0.064
    elif self.snake_type == "REU":
        module_length = 0.050
    if abs(A_even) > A_transition+0.2:
        p = module_length/((np.abs(A_even)/2/np.sin(np.abs(wS_even)))**2+1)/np.abs(wS_even)
        r = np.abs(A_even)/2/np.sin(np.abs(wS_even))*p
        offset_p = -p*0.25
        offset_r = r*10
        wS_bump = p/((r+offset_r)**2+(p+offset_p)**2)*module_length * np.sign(wS_even)
        A_bump = 2*(r+offset_r)/(p+offset_p)*np.sin(np.abs(wS_bump)) * np.sign(A_even)
    else:
        A_bump = A_even
        wS_bump = wS_even

    A_set_spiraling = [A_bump * A_1_multiplier, A_bump *  A_2_multiplier]
    wS_set_spiraling = [wS_bump, wS_bump]
    
    A_set = [A_1, A_2]
    wS_set = [wS_1, wS_2]
    #n_window = mu
    # Tunable window parameters (allow teleop / YAML overrides).
    # Defaults preserve the original snakes_on_pipes behavior.
    m = float(params.get('m', 50.0))        # sharpness of the exp/sigmoid window
    sig = float(params.get('sig', 0.05))    # variance of the gaussian window
    T = float(params.get('T', 0.25))        # period of the sinusoidal window
    for i in range(self.num_modules):
        offset = 0
        offset_hook = 0
        
        if i%2 ==0:
            offset = np.pi
        else:
            offset = -np.pi/2

        offset_hook = np.sin(phi_0 + wS_even * i + wT_even * t + offset)        

        target_angles[i] = self.amplitude_reduced(i, A_set, m, mu, sig) * np.sin(
            self.parameter_windowed(i, wS_set, mu, m) * i + wT_even * t + offset
        ) + offset_hook * A_even * self.gaussian_window(i / 15, mu / 15, sig)

        """vvvv BLock of code for spiraling gait vvvv"""
        cont_offset = (wS_even-wS_bump)*(mu/15-T/2)*self.num_modules

        target_angles_spiraling[i] = self.amplitude_reduced(i, A_set_spiraling, m, mu, sig) * np.sin(
            self.parameter_windowed(i, wS_set_spiraling, mu, m) * i + cont_offset + wT_even * t + offset
        ) + offset_hook * A_even * self.gaussian_window(i / 15, mu / 15, sig)

        target_angles[i] = target_angles[i]*(1-self.sinus_window(i/(self.num_modules-1),s_0,T)) + target_angles_spiraling[i]*self.sinus_window(i/(self.num_modules-1),s_0,T)
        
        target_angles[i] = min(max(target_angles[i],-np.pi/2),np.pi/2)

    # flipping angles to account for the harware flip of the axes    
    target_angles[2::4] *= -1
    target_angles[3::4] *= -1

    # target_angles[len(target_angles) - 2:len(target_angles)] *= 0.75      # this can help the snake smooth its gait when tightened around a pole
    # target_angles[0:2] *= 0.75                                            # this can help the snake smooth its gait when tightened around a pole

    if debug_print:
        param_names = ["A1_multiplier", "A2_multiplier", "A_even", "wS_even", "wT_even", "mu", "phi_0", "s_0", "tightness", "speed_multiplier"]
        param_values = [A_1_multiplier, A_2_multiplier, A_even, wS_even, wT_even, mu, phi_0, s_0, tightness, speed_multiplier]
        param_dict = dict(zip(param_names, param_values))
        print(param_dict)

    if not compute:
        # Return the parameters used (handy for printing / logging)
        out = deepcopy(self.current_gait_params)
        out.update({
            "A_1_multiplier": A_1_multiplier,
            "A_2_multiplier": A_2_multiplier,
            "mu": mu,
            "phi_0": phi_0,
            "s_0": s_0,
            "speed_multiplier": speed_multiplier,
        })
        return out

    return target_angles


def gaussian_window(self, x, mu, sigma):
        """
        This function evaluates the input variable x on a window function shaped as a Gaussian
        function with mean mu and standard deviation sigma.
        """
        return np.exp(-((x - mu)/sigma) ** 2 /2)
 
def sinus_window(self, x, x0, T):
    """
    This function evaluates the input variable x on a window function with a sinusoidal shape defined
    between x0 and x0+T. The window is 0 outside of this range.
    """
    temp = (x-x0)*2*np.pi/T
    coef = 1

    # uncomment the next three lines to compensate in the sinusoidal window for the position error introduced
    if temp > -np.pi:
        temp = -np.pi+(temp+np.pi)*0.85
        coef = 0.25
    if (x0-T/2)<0.5:
        return max(((x0-T/2)**2) *8,0) * np.sin(min(max(temp,-2*np.pi),0))*coef    # single wave
    elif (x0-T/2)>0.5:
        return max(((1-(x0-T/2))**2)*8,0) * np.sin(min(max(temp,-2*np.pi),0))*coef # single wave
    else:
        return np.sin(min(max(temp,-2*np.pi),0))*coef               
        # return np.sin(temp)                       # continuous sinus wave
    
def exp_window(self, x, m, n_start, n_end):
    """
    This function evaluates the input variable x on a window function approximating a rectangle
    using an exponential function. The window is defined between n_start and n_end. The window
    is 0 outside of this range. The slope of the transition is defined by m.
    """
    return (1 / (1 + np.exp(-m * min((x - n_start),500))) + 1 / (1 + np.exp(m * min((x - n_end),500))) -1)
    
def parameter_windowed(self, i, A_set, mu, m_slope):
    """
    This function computes the active gait amplitude at module i based on the set of gait
    parameters A_set and the two windows before and after module n. 
    """
    return A_set[0]* self.exp_window(i, m_slope, -1, mu) + A_set[1]* self.exp_window(i, m_slope, mu, 16)
    
def amplitude_reduced(self, i, A_set, m_slope, mu, sigma):
    return self.parameter_windowed(i, A_set, mu, m_slope) * (1-self.gaussian_window(i/(self.num_modules-1), mu/15, sigma))
    
def amplitude_reduced_sinus(self, i, A_set, n, m_slope, x0, T):
    return self.parameter_windowed(i, A_set, n, m_slope) * (1-self.sinus_window(i/(self.num_modules-1), x0, T))
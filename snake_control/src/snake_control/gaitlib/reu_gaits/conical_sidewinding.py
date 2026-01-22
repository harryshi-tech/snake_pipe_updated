import numpy as np
from copy import deepcopy

def conical_sidewinding(self, t=0, params=None):

    params = {} if params is None else params

    self.current_gait = "conical_sidewinding"

    # Update the current parameters if params is not empty
    self.current_gait_params = self.update_params(
        self.default_gait_params.get(self.current_gait), params
    )

    N = self.num_modules
    
    alpha = np.zeros(N)

    gait_params = deepcopy(self.current_gait_params)

    for n in range(N):
        if n%2==0:
            A_even = self.current_gait_params['A_even'] + self.current_gait_params.get('slope',0)*n
            gait_params['A_even'] = np.clip(A_even, self.current_gait_params['A_even'] - self.current_gait_params['A_even_tol'], 
            self.current_gait_params['A_even'] + self.current_gait_params['A_even_tol'])
        
        else:
            A_odd = self.current_gait_params['A_odd'] + self.current_gait_params.get('slope',0)*n
            gait_params['A_odd'] = np.clip(A_odd, self.current_gait_params['A_odd'] - self.current_gait_params['A_odd_tol'], 
            self.current_gait_params['A_odd'] + self.current_gait_params['A_odd_tol'])
        
        alpha[n] = self.compound_serpenoid(t, n, gait_params)
    
    return alpha

    return target_angles


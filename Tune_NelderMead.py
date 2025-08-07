import numpy as np
#import json
from scipy.optimize import minimize

#import torch
#from sbi import utils as utils
#from sbi.utils.sbiutils import seed_all_backends
#from sbi.inference.base import infer
#from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
##from sbi.simulators.simutils import tqdm_joblib
##from tqdm.auto import tqdm
##from joblib import Parallel, delayed
##import pickle
#from multiprocessing import Pool

#   config_file = './svzerod_tuning.json'
#   f = open(config_file)
#   config = json.load(f)
#   params = np.genfromtxt('optParams.txt')

#   for i in range(100):
#       model = closed_loop_model.svZeroD_ClosedLoop(config)
#       frozen_param_idxs, variable_param_idxs = model.read_frozen_params('frozen_parameters.csv')
#       results = model.run_with_frozen_params(params[variable_param_idxs])


#   model = closed_loop_model.svZeroD_ClosedLoop(config)
#   for i in range(100):
#       frozen_param_idxs, variable_param_idxs = model.read_frozen_params('frozen_parameters.csv')
#       results = model.run_with_frozen_params(params[variable_param_idxs])

class Tune_NelderMead:

#   base_config = None
#   model_metadata = None
#   frozen_param_idxs = None
#   variable_param_idxs = None
#   frozen_param_values = None
    model = None
    num_restarts = None
    adaptive = False
    max_iter = None
    num_params = None
    num_results = None
    bounds = None
    initial_guess = None

#   def __init__(self, config_file):
#       f = open(config_file)
#       self.base_config = json.load(f)
#       model = closed_loop_model.svZeroD_ClosedLoop(self.base_config)
#       self.model_metadata = model.get_data_members()
#       self.frozen_param_idxs, self.variable_param_idxs, self.frozen_param_values = model.read_frozen_params('frozen_parameters.csv')
#       self.num_params = model.get_num_parameters()
#       self.num_results = model.get_num_results()

    def __init__(self, model, optimization_params):
        self.num_restarts = optimization_params["num_restarts"]
        self.max_iter = optimization_params["max_iterations"]
        self.convergence_tol = optimization_params.get("convergence_tol", 0.1)
        self.model = model
        self.num_params = model.num_parameters()
        self.num_results = model.num_results()
        self.bounds = model.parameter_limits_tuples()
        self.options = {}
        #self.options["bounds"] = self.bounds
        self.options["maxiter"] = self.max_iter
        self.options["adaptive"] = optimization_params.get("adaptive", False)
        self.options["disp"] = optimization_params.get("verbose", True)
        self.print_info()

    def print_info(self):
        print("--------------------------------")
        print("Running Nelder Mead optimization")
        print(f"Number of restarts: {self.num_restarts}")
        print(f"Max iterations: {self.max_iter}")
        print(f"Number of parameters: {self.num_params}")
        print("--------------------------------")

    def set_initial_guess(self, initial_params):
        if len(initial_params) != self.num_params:
            raise RuntimeError("len(initial_guess) != self.num_params")
        self.initial_guess = initial_params

    def run(self):
        init_guess = self.initial_guess
        for i_restart in range(self.num_restarts):
            print(f"\nRunning optimization round {i_restart+1}/{self.num_restarts} \n")
            print("Initial guess:")
            print(init_guess)
            res = minimize(self.model.evaluate_error, init_guess, method='Nelder-Mead', 
                           tol=self.convergence_tol, bounds=self.bounds, options=self.options)
            init_guess = res.x
        return res.x

#   def evaluate(self, variable_params):
#       full_param_vector = np.zeros(self.num_params)
#       full_param_vector[self.frozen_param_idxs] = self.frozen_param_values
#       full_param_vector[self.variable_param_idxs] = variable_params
#       new_config = self.update_config_params(self.base_config, self.model_metadata, full_param_vector) 
#       model = closed_loop_model.svZeroD_ClosedLoop(new_config)
#       try:
#           results = model.run_model()
#           return results
#       except:
#           print('Invalid result for parameters: ', variable_params)
#           return np.empty(self.num_results)*np.nan

#   def evaluate_simple(self, params):
#       model = closed_loop_model.svZeroD_ClosedLoop(self.base_config)
#       frozen_param_idxs, variable_param_idxs, _ = model.read_frozen_params('frozen_parameters.csv')
#       try:
#           results = model.run_with_frozen_params(params)
#           return results
#       except:
#           print('Invalid result for parameters: ', params)
#           return np.empty(self.num_results)*np.nan

 #  def create_prior(self):
 #      model = closed_loop_model.svZeroD_ClosedLoop(self.base_config)
 #      all_parameter_limits = model.parameter_limits()
 #      lower_lims = all_parameter_limits[np.arange(0,2*model.get_num_parameters(),2)]
 #      upper_lims = all_parameter_limits[np.arange(0,2*model.get_num_parameters(),2)+1]
 #      prior = utils.BoxUniform(low=torch.tensor(lower_lims[self.variable_param_idxs]), high=torch.tensor(upper_lims[self.variable_param_idxs]))
 #      return prior

 #  def run_simulations(self, num_procs, parameter_samples):
 #      p = Pool(num_procs)
 #      all_results = p.map(self.evaluate, parameter_samples)
 #      #all_results = p.map(self.evaluate_simple, parameter_samples)
 #      return torch.tensor(np.array(all_results))

#   def update_config_params(self, original_config, metadata, params):
#       new_config = original_config.copy()
#       for i, bc_idx in enumerate(metadata['idxs_corBC_l']):
#           bc_values = new_config['boundary_conditions'][bc_idx]['bc_values'] 
#           bc_values['Ra'] = metadata['Ra_l_base'][i]*params[26]
#           bc_values['Ram'] = metadata['Ram_l_base'][i]*params[26]
#           bc_values['Rv'] = metadata['Rv_l_base'][i]*params[27]
#           bc_values['Ca'] = metadata['Ca_l_base'][i]*params[29]
#           bc_values['Cim'] = metadata['Cim_l_base'][i]*params[28]
#       
#       for i, bc_idx in enumerate(metadata['idxs_corBC_r']):
#           bc_values = new_config['boundary_conditions'][bc_idx]['bc_values'] 
#           bc_values['Ra'] = metadata['Ra_r_base'][i]*params[26]
#           bc_values['Ram'] = metadata['Ram_r_base'][i]*params[26]
#           bc_values['Rv'] = metadata['Rv_r_base'][i]*params[27]
#           bc_values['Ca'] = metadata['Ca_r_base'][i]*params[31]
#           bc_values['Cim'] = metadata['Cim_r_base'][i]*params[30]
#       
#       for i, bc_idx in enumerate(metadata['idxs_RCRBC']):
#           bc_values = new_config['boundary_conditions'][bc_idx]['bc_values'] 
#           bc_values['Rp'] = metadata['Rp_rcr_base'][i]*params[32]
#           bc_values['C'] = metadata['C_rcr_base'][i]*params[33]
#           bc_values['Rd'] = metadata['Rd_rcr_base'][i]*params[32]
#   	
#       heart_params = new_config['closed_loop_blocks'][0]['parameters']
#       heart_params['Tsa']  = params[0]
#       heart_params['tpwave']  = params[1]
#       heart_params['Erv_s']  = params[2]
#       heart_params['Elv_s']  = params[3]
#       heart_params['iml']  = params[4]
#       heart_params['imr']  = params[34]
#       heart_params['Lra_v']  = params[7]/metadata['pConv']; 
#       heart_params['Rra_v']  = params[8]/metadata['pConv']; 
#       heart_params['Lrv_a']  = params[5]/metadata['pConv']; 
#       heart_params['Rrv_a']  = metadata['Rrv_base']*params[6]/metadata['pConv']
#       heart_params['Lla_v'] = params[9]/metadata['pConv']
#       heart_params['Rla_v'] = params[10]/metadata['pConv']
#       heart_params['Llv_a'] = params[12]/metadata['pConv']
#       heart_params['Rlv_ao'] = metadata['Rlv_base']*params[11]/metadata['pConv']
#       heart_params['Vrv_u'] = params[13]
#       heart_params['Vlv_u'] = params[14]
#       heart_params['Rpd'] = metadata['Rpd_base']*params[15]/metadata['pConv']
#       heart_params['Cp'] = params[16]
#       heart_params['Cpa'] = params[17]
#       heart_params['Kxp_ra'] = params[18]
#       heart_params['Kxv_ra'] = params[19]
#       heart_params['Kxp_la'] = params[22]
#       heart_params['Kxv_la'] = params[23]
#       heart_params['Emax_ra'] = params[20]
#       heart_params['Emax_la'] = params[24]
#       heart_params['Vaso_ra'] = params[21]
#       heart_params['Vaso_la'] = params[25]
#       
#       return new_config


if __name__ == "__main__":

    config_file = './svzerod_tuning.json'
    sbi_coronary = SBI_ClosedLoopCoronary(config_file)
#   prior = sbi_coronary.create_prior()
#   num_samples = 20
#   param_samples = prior.sample((num_samples,))
#   print(param_samples)

#   num_procs = 2
#   all_results = sbi_coronary.run_simulations(num_procs, param_samples)
#   print(all_results)
#   print(len(all_results))




#simulator, prior = prepare_for_sbi(model.run_with_frozen_params, prior)
#num_samples = 5
#params, results = create_samples_and_results(num_samples, simulator, prior)

#posterior = infer(model.run_with_frozen_params, prior, method="SNPE", num_simulations=1000)
#params, results = simulate_for_sbi(model.run_with_frozen_params, proposal = prior, num_simulations = 100, num_workers = 2)

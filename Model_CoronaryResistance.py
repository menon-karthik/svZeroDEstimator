import pysvzerod
import numpy as np
import json
import matplotlib.pyplot as plt
import copy

np.set_printoptions(formatter={'float': lambda x: format(x, '6.5E')})

class Model_CoronaryResistance:

    # Private members
    __config = None
    __solver = None
    __frozen_param_idxs = None
    __frozen_param_values = None

    # Targets
#   targets = {}
#   targets_values = None
#   #targets_idxs = None # Indices for non-NaN targets
    total_target_flow = 0.0
    target_flows = None
    outlet_names = None
    target_flow_fracs = None

    # Model-specific members
    pConv = 1333.22
    names_corBC_l = []
    names_corBC_r = []
    branchnames_corBC_l = []
    branchnames_corBC_r = []
#   names_RCRBC = []
    idxs_corBC_l = []
    idxs_corBC_r = []
#   idxs_RCRBC = []
    n_corBC_l = 0
    n_corBC_r = 0
    n_corBC = 0
#   n_RCRBC = 0
    Ra_l_base = []
    Ram_l_base = []
    Rv_l_base = []
    Ca_l_base = []
    Cim_l_base = []
    Ra_r_base = []
    Ram_r_base = []
    Rv_r_base = []
    Ca_r_base = []
    Cim_r_base = []
#   Rp_rcr_base = []
#   C_rcr_base = []
#   Rd_rcr_base = []
#   Rrv_base = None
#   Rlv_base = None
#   Rpd_base = None
    steps_per_cycle = None
    num_cycles = None
    bc_vessel_map = {}

    # Perfusion data
    use_perfusion = False
    perfusion_mbf = {}
    perfusion_vol = {}

    R_total_inv_base = 0.0
    Q_lca_names = []
    Q_rca_names = []
    result_weights = None

    # Resistance scaling
    #self.scaled = False
    R_scaling = 0.0
    R_scaling_history = []
    # ------------------------------------------------------------
    
    def __init__(self, config, target_file, perfusion_filename = None):
        if isinstance(config, dict):
            self.__config = config
        else:
            f = open(config)
            self.__config = json.load(f)
        
        self.__solver = pysvzerod.Solver(self.__config)
        
        if perfusion_filename != None:
            self.perfusion_mbf, self.perfusion_vol = self.read_perfusion_file(perfusion_filename)
            self.use_perfusion = True

        self.read_target_flows(target_file)

        self.setup_model(self.__config)
        
    # ------------------------------------------------------------
    
    def read_perfusion_file(self, perfusion_filename):
        perf_names = np.genfromtxt(perfusion_filename, dtype = 'U', skip_header=1, usecols=0)
        perf_mbf = np.genfromtxt(perfusion_filename, dtype = 'float', skip_header=1, usecols=1)
        perf_vol = np.genfromtxt(perfusion_filename, dtype = 'float', skip_header=1, usecols=2)
        perf_mbf_dict = dict(zip(perf_names, perf_mbf))
        perf_vol_dict = dict(zip(perf_names, perf_vol))
        return perf_mbf_dict, perf_vol_dict
        
    # ------------------------------------------------------------
   
    def write_branch_file(self):
        try:
            with open("distalR_tuning_branches.txt", 'w') as branch_file:
                for block_name in self.names_corBC_l:
                    branch_file.write(block_name + "\n")
                for block_name in self.names_corBC_r:
                    branch_file.write(block_name + "\n")
        except Exception as e:
            print(f"Error writing to file: {e}")
        
    # ------------------------------------------------------------

    def read_target_flows(self, target_flows_filename):
        self.outlet_names = np.genfromtxt(target_flows_filename, dtype = 'U', skip_header=0, usecols=0)
        self.target_flows = np.genfromtxt(target_flows_filename, dtype = 'float', skip_header=0, usecols=1)
        self.total_target_flow = np.sum(self.target_flows)
        
    # ------------------------------------------------------------

    def discard_target_flows_without_perfusion_data(self):
        if self.use_perfusion:
            idxs_to_delete = []
            for i, name in enumerate(self.outlet_names):
                if self.perfusion_mbf[name] <= 0.0:
                    idxs_to_delete.append(i)
            self.target_flows = np.delete(self.target_flows, idxs_to_delete)
            self.outlet_names = np.delete(self.outlet_names, idxs_to_delete)
            self.total_target_flow = np.sum(self.target_flows)
        
    # ------------------------------------------------------------

    def rearrange_targets(self):
        targets_copy = copy.deepcopy(self.target_flows)
        outlet_names_copy = copy.deepcopy(self.outlet_names)

        # Rearrange left coronary artery targets
        for i, name in enumerate(self.branchnames_corBC_l):
            print(f"Branch name: {name}")
            try:
                idx = np.where(outlet_names_copy == name)[0][0]
                print(f"Found in targets at idx: {idx}")
                self.target_flows[i] = targets_copy[idx]
                self.outlet_names[i] = outlet_names_copy[idx]
                print(f"Rearrange: {i} {idx}")
            except ValueError:
                raise RuntimeError(f"Error: Could not find {name} in outlet_names.")

        # Rearrange right coronary artery targets
        offset = len(self.branchnames_corBC_l)
        for i, name in enumerate(self.branchnames_corBC_r):
            print(f"Branch name: {name}")
            try:
                idx = np.where(outlet_names_copy == name)[0][0]
                print(f"Found in targets at idx: {idx}")
                self.target_flows[offset + i] = targets_copy[idx]
                self.outlet_names[offset + i] = outlet_names_copy[idx]
                print(f"Rearrange: {offset + i} {idx}")
            except ValueError:
                raise RuntimeError(f"Error: Could not find {name} in outlet_names.")
    
    # ------------------------------------------------------------

    def get_specified_parameter(self, specifier):
        if specifier == "RScaling":
            return self.R_scaling
        elif specifier == "RScaling_history":
            return self.R_scaling_history
        else:
            raise RuntimeError("Invalid specifier in Model_CoronaryResistance -> get_specified_parameter.")

    # ------------------------------------------------------------
    
    def setup_model(self, config):

        self.steps_per_cycle = config["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"]
        self.num_cycles = config["simulation_parameters"]["number_of_cardiac_cycles"]
        
        # Find name of vessel corresponding to each outlet boundary condition
        for vessel in config["vessels"]:
            if "boundary_conditions" in vessel:
                if "outlet" in vessel["boundary_conditions"]:
                    self.bc_vessel_map[vessel["boundary_conditions"]["outlet"]] = vessel["vessel_name"]
       
        # Save baseline coronary resistances, totoal baseline resistance, and names of flow variables
        # If using perfusion_data.dat, this data is only saved for branches with MBF > 0 
        # (i.e. branches for which there is perfusion data)
        for i, bc in enumerate(config["boundary_conditions"]):
            if "BC_lca" in bc["bc_name"]:
                branch_name = bc["bc_name"][3:]
                if (self.use_perfusion == False or self.perfusion_mbf[branch_name] > 0.0):
                    # BC and vessel name
                    self.names_corBC_l.append(bc["bc_name"])
                    self.idxs_corBC_l.append(i)
                    self.branchnames_corBC_l.append(branch_name)
                    # BC values
                    self.Ra_l_base.append(bc["bc_values"]["Ra"])
                    self.Ram_l_base.append(bc["bc_values"]["Ram"])
                    self.Rv_l_base.append(bc["bc_values"]["Rv"])
                    self.Ca_l_base.append(bc["bc_values"]["Ca"])
                    self.Cim_l_base.append(bc["bc_values"]["Cim"])
                    self.R_total_inv_base += 1.0/(self.Ra_l_base[-1]+self.Ram_l_base[-1]+self.Rv_l_base[-1])
                    # Name of solution variable corresponding to flow in this boundary condition
                    self.Q_lca_names.append("flow:"+self.bc_vessel_map[bc["bc_name"]]+":"+bc["bc_name"])
            elif "BC_rca" in bc["bc_name"]:
                branch_name = bc["bc_name"][3:]
                if (self.use_perfusion == False or self.perfusion_mbf[branch_name] > 0.0):
                    # BC and vessel name
                    self.names_corBC_r.append(bc["bc_name"])
                    self.idxs_corBC_r.append(i)
                    self.branchnames_corBC_r.append(branch_name)
                    # BC values
                    self.Ra_r_base.append(bc["bc_values"]["Ra"])
                    self.Ram_r_base.append(bc["bc_values"]["Ram"])
                    self.Rv_r_base.append(bc["bc_values"]["Rv"])
                    self.Ca_r_base.append(bc["bc_values"]["Ca"])
                    self.Cim_r_base.append(bc["bc_values"]["Cim"])
                    self.R_total_inv_base += 1.0/(self.Ra_r_base[-1]+self.Ram_r_base[-1]+self.Rv_r_base[-1])
                    # Name of solution variable corresponding to flow in this boundary condition
                    self.Q_rca_names.append("flow:"+self.bc_vessel_map[bc["bc_name"]]+":"+bc["bc_name"])

        self.n_corBC_l = len(self.names_corBC_l)
        self.n_corBC_r = len(self.names_corBC_r)
        self.n_corBC = self.n_corBC_l + self.n_corBC_r

        # Write a file with order of coronary branches to specify order of parameters
        self.write_branch_file()

        print(f"Total baseline coronary resistance = {1.0 / self.R_total_inv_base}")

        # Discard targets without perfusion data if required
        self.discard_target_flows_without_perfusion_data()

        # Make sure number of targets is the same as the number of saved outlets (with or without perfusion data)
        if self.n_corBC != len(self.target_flows):
            print(f"Number of targets = {len(self.target_flows)}; Number of saved coronary outlets = {self.n_corBC}.")
            raise RuntimeError("Number of targets is not the same as number of saved coronary outlets.")

        # Rearrange targets to match the order of names in Q_lca/rca_names or names_corBC_l/r
        self.rearrange_targets()

        # Save target flow fractions, weights and standard deviation
        self.target_flow_fracs = self.target_flows/self.total_target_flow
        self.result_weights = np.ones(len(self.target_flows))
    
    # ------------------------------------------------------------
    
    def num_parameters(self):
        return self.n_corBC_l + self.n_corBC_r
    
    # ------------------------------------------------------------
    
    def num_results(self):
        return self.n_corBC_l + self.n_corBC_r
    
    # ------------------------------------------------------------
    
    def run_model(self):
        self.__solver.run()
        results = self.post_process() 
        return results
    
    # ------------------------------------------------------------
    
    def run_with_params(self, new_params):
        self.update_model_params(new_params)
        self.__solver.run()
        results = self.post_process() 
        return results
    
    # ------------------------------------------------------------
    
    def evaluate_error(self, params):
        results = self.run_with_params(params)
        error = self.weighted_mse_error(results)
        return error
    
    # ------------------------------------------------------------

    def weighted_mse_error(self, results):
        weights = self.get_result_weights()
        error = np.mean(np.divide(np.square(np.divide((self.target_flow_fracs - results), self.target_flow_fracs)), weights))
        print(error)
        return error
    
    # ------------------------------------------------------------

    def update_model_params(self, params):
        R_total_inv = 0.0

        # Calculate scaling
        for i in range(self.n_corBC_l):
            outlet_R = (self.Ra_l_base[i] + self.Ram_l_base[i] + self.Rv_l_base[i]) * params[i]
            R_total_inv += 1.0 / outlet_R

        for i in range(self.n_corBC_r):
            idx = self.n_corBC_l + i
            outlet_R = (self.Ra_r_base[i] + self.Ram_r_base[i] + self.Rv_r_base[i]) * params[idx]
            R_total_inv += 1.0 / outlet_R

        self.R_scaling = R_total_inv / self.R_total_inv_base
        self.R_scaling_history.append(self.R_scaling)

        R_total_inv = 0.0

        coronary_params = np.zeros(5)
        for i, block_name in enumerate(self.names_corBC_l):
            coronary_params[0] = self.Ra_l_base[i]*params[i]*self.R_scaling
            coronary_params[1] = self.Ram_l_base[i]*params[i]*self.R_scaling
            coronary_params[2] = self.Rv_l_base[i]*params[i]*self.R_scaling
            coronary_params[3] = self.Ca_l_base[i]
            coronary_params[4] = self.Cim_l_base[i]
            self.__solver.update_block_params(block_name, coronary_params)
            R_total_inv += 1.0 / sum(coronary_params[0:3])
        
        for i, block_name in enumerate(self.names_corBC_r):
            idx = self.n_corBC_l + i
            coronary_params[0] = self.Ra_r_base[i]*params[idx]*self.R_scaling
            coronary_params[1] = self.Ram_r_base[i]*params[idx]*self.R_scaling
            coronary_params[2] = self.Rv_r_base[i]*params[idx]*self.R_scaling
            coronary_params[3] = self.Ca_r_base[i]
            coronary_params[4] = self.Cim_r_base[i]
            self.__solver.update_block_params(block_name, coronary_params)
            R_total_inv += 1.0 / sum(coronary_params[0:3])

        print(f"Total assigned coronary resistance = {1.0 / R_total_inv}")
    
    # ------------------------------------------------------------
    
    def update_json_config(self, params, r_scale):

        new_config = self.__config.copy()
        
        for i, bc_idx in enumerate(self.idxs_corBC_l):
            bc_values = new_config['boundary_conditions'][bc_idx]['bc_values'] 
            bc_values['Ra'] = self.Ra_l_base[i]*params[i]*r_scale
            bc_values['Ram'] = self.Ram_l_base[i]*params[i]*r_scale
            bc_values['Rv'] = self.Rv_l_base[i]*params[i]*r_scale
            bc_values['Ca'] = self.Ca_l_base[i]
            bc_values['Cim'] = self.Cim_l_base[i]
        
        for i, bc_idx in enumerate(self.idxs_corBC_r):
            idx = self.n_corBC_l + i
            bc_values = new_config['boundary_conditions'][bc_idx]['bc_values'] 
            bc_values['Ra'] = self.Ra_r_base[i]*params[idx]*r_scale
            bc_values['Ram'] = self.Ram_r_base[i]*params[idx]*r_scale
            bc_values['Rv'] = self.Rv_r_base[i]*params[idx]*r_scale
            bc_values['Ca'] = self.Ca_r_base[i]
            bc_values['Cim'] = self.Cim_r_base[i]
       
        return new_config
        
    # ------------------------------------------------------------

    def get_result_weights(self):
        return self.result_weights
    
    # ------------------------------------------------------------

    def parameter_limits_tuples(self):
        limits = self.parameter_limits()
        limits_tuples = []
        for param_idx in range(self.num_parameters()):
            limits_tuples.append((limits[2*param_idx], limits[2*param_idx+1]))
        return limits_tuples

    # ------------------------------------------------------------

    def parameter_limits(self):
        limits = np.zeros(2*self.num_parameters())
        for param_idx in range(self.num_parameters()):
            limits[2*param_idx] = 0.5
            limits[2*param_idx+1] = 2.0
        return limits
    
    # ------------------------------------------------------------
    
    def post_process(self):
        
        results = np.full(self.num_results(), np.nan)
        Q_cor = 0.0

        # Left coronary flow
        for i in range(self.n_corBC_l):
            var = self.__solver.get_single_result(self.Q_lca_names[i])
            results[i] = np.mean(var[-self.steps_per_cycle:])
            Q_cor += results[i]
        
        # Right coronary flow
        for i in range(self.n_corBC_r):
            var = self.__solver.get_single_result(self.Q_rca_names[i])
            results[self.n_corBC_l + i] = np.mean(var[-self.steps_per_cycle:])
            Q_cor += results[self.n_corBC_l + i]

        # Convert to flow fractions
        results /= Q_cor

        print(f"Total coronary flow = {Q_cor}")

        return results
    
    # ------------------------------------------------------------
    
    def write_results_and_targets(self, results, savepath = '.'):
        
        write_data = np.array(list(zip(self.outlet_names, self.target_flow_fracs, results)), 
                              dtype=[('target_names','U16'),('target_values',float),('results',float)])
        
        savename = savepath + '/results_targets.dat'

        np.savetxt(savename, write_data, fmt = '%-16s %-18.10e %-18.10e', header='Name, Target, Result')
    
    # ------------------------------------------------------------
    
    def plot_0D_quantity(self, name, path = None, limits = None):
        var = self.__solver.get_single_result(name)
        times = self.__solver.get_times()
        plt.figure()
        plt.plot(times,var)
        if (limits is not None):
            plt.xlims(xlims)
        if (path == None):
            savename = name + ".pdf"
        else:
            savename = path + name + ".pdf"
        plt.savefig(savename)
        plt.close()

# ------------------------------------------------------------
# ------------------------------------------------------------

if __name__ == "__main__":

    data_folder = 'from_flow_mpi/afterDistalR/' 
    f = open(data_folder+'../afterClosedLoop/svzerod_updateconfig_closedloop.json')
    original_config = json.load(f)
    
    model_dummy = Model_CoronaryResistance(original_config, data_folder+'target_flows.dat', data_folder+'perfusion_volumes.dat')
    params = np.genfromtxt(data_folder+'optParams.txt')
    model_dummy.update_model_params(params)
    results = model_dummy.run_model()
    model_dummy.write_results_and_targets(results)
    r_scaling = model_dummy.get_specified_parameter("RScaling")
    new_config = model_dummy.update_json_config(params, r_scaling)
    with open(data_folder+'svzerod_updateconfig_distalR.json', 'w') as f:
            json.dump(new_config, f, indent=4)


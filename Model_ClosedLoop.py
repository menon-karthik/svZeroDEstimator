import pysvzerod
import numpy as np
import json
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': lambda x: format(x, '6.5E')})

class Model_ClosedLoop:

    # Private members
    __config = None
    __solver = None
    __frozen_param_idxs = None
    __frozen_param_values = None

    # Targets/results
    targets = {}
    targets_values = None
    targets_idxs = None # Indices for non-NaN targets
    results_scale = None

    # Model-specific members
    p_conv = 1333.22
    names_corBC_l = []
    names_corBC_r = []
    names_RCRBC = []
    idxs_corBC_l = []
    idxs_corBC_r = []
    idxs_RCRBC = []
    n_corBC_l = 0
    n_corBC_r = 0
    n_corBC = 0
    n_RCRBC = 0
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
    Rp_rcr_base = []
    C_rcr_base = []
    Rd_rcr_base = []
    Rrv_base = None
    Rlv_base = None
    Rpd_base = None
    steps_per_cycle = None
    num_cycles = None
    bc_vessel_map = {}
    # ------------------------------------------------------------
    
    def __init__(self, config, data = None):
        if isinstance(config, dict):
            self.__config = config
        else:
            f = open(config)
            self.__config = json.load(f)
        
        self.__solver = pysvzerod.Solver(self.__config)
        
        if (data == None):
            self.setup_model(self.__config)
        else:
            self.assign_model(data)

        self.check_data()
    
    # ------------------------------------------------------------
    
    def check_data(self):
        if self.num_parameters() != len(self.parameter_names()):
            raise RuntimeError("Number of parameters is different from number of parameter names.")
        if self.num_results() != len(self.results_names()):
            raise RuntimeError("Number of results is different from number of results names.")

    # ------------------------------------------------------------
    
    def setup_model(self, config):

        self.steps_per_cycle = config["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"]
        self.num_cycles = config["simulation_parameters"]["number_of_cardiac_cycles"]
        
        for i, bc in enumerate(config["boundary_conditions"]):
            if "BC_lca" in bc["bc_name"]:
                self.names_corBC_l.append(bc["bc_name"])
                self.idxs_corBC_l.append(i)
                self.Ra_l_base.append(bc["bc_values"]["Ra"])
                self.Ram_l_base.append(bc["bc_values"]["Ram"])
                self.Rv_l_base.append(bc["bc_values"]["Rv"])
                self.Ca_l_base.append(bc["bc_values"]["Ca"])
                self.Cim_l_base.append(bc["bc_values"]["Cim"])
            elif "BC_rca" in bc["bc_name"]:
                self.names_corBC_r.append(bc["bc_name"])
                self.idxs_corBC_r.append(i)
                self.Ra_r_base.append(bc["bc_values"]["Ra"])
                self.Ram_r_base.append(bc["bc_values"]["Ram"])
                self.Rv_r_base.append(bc["bc_values"]["Rv"])
                self.Ca_r_base.append(bc["bc_values"]["Ca"])
                self.Cim_r_base.append(bc["bc_values"]["Cim"])
            if "BC_RCR" in bc["bc_name"]:
                self.names_RCRBC.append(bc["bc_name"])
                self.idxs_RCRBC.append(i)
                self.Rp_rcr_base.append(bc["bc_values"]["Rp"])
                self.C_rcr_base.append(bc["bc_values"]["C"])
                self.Rd_rcr_base.append(bc["bc_values"]["Rd"])

        self.n_corBC_l = len(self.names_corBC_l)
        self.n_corBC_r = len(self.names_corBC_r)
        self.n_corBC = self.n_corBC_l + self.n_corBC_r
        self.n_RCRBC = len(self.names_RCRBC)

        if len(config["closed_loop_blocks"]) != 1:
            print(len(config["closed_loop_blocks"]))
            raise RuntimeError("len(config[\"closed_loop_blocks\"]) != 1")
        closed_loop_block = config["closed_loop_blocks"][0]
        self.Rrv_base = closed_loop_block["parameters"]["Rrv_a"]
        self.Rlv_base = closed_loop_block["parameters"]["Rlv_ao"]
        self.Rpd_base = closed_loop_block["parameters"]["Rpd"]
        
        for vessel in config["vessels"]:
            if "boundary_conditions" in vessel:
                if "outlet" in vessel["boundary_conditions"]:
                    self.bc_vessel_map[vessel["boundary_conditions"]["outlet"]] = vessel["vessel_name"]
        
        weights = self.get_result_weights()[self.targets_idxs]
        std = self.get_result_std()[self.targets_idxs]
        self.results_scale = np.multiply(np.square(std), weights)

    # ------------------------------------------------------------

    def assign_model(self, data):
        self.names_corBC_l = data['names_corBC_l']
        self.names_corBC_r = data['names_corBC_r']
        self.names_RCRBC = data['names_RCRBC']
        self.idxs_corBC_l = data['idxs_corBC_l']
        self.idxs_corBC_r = data['idxs_corBC_r']
        self.idxs_RCRBC = data['idxs_RCRBC']
        self.n_corBC_l = data['n_corBC_l']
        self.n_corBC_r = data['n_corBC_r']
        self.n_corBC = data['n_corBC']
        self.n_RCRBC = data['n_RCRBC']
        self.Ra_l_base = data['Ra_l_base']
        self.Ram_l_base = data['Ram_l_base']
        self.Rv_l_base = data['Rv_l_base']
        self.Ca_l_base = data['Ca_l_base']
        self.Cim_l_base = data['Cim_l_base']
        self.Ra_r_base = data['Ra_r_base']
        self.Ram_r_base = data['Ram_r_base']
        self.Rv_r_base = data['Rv_r_base']
        self.Ca_r_base = data['Ca_r_base']
        self.Cim_r_base = data['Cim_r_base']
        self.Rp_rcr_base = data['Rp_rcr_base']
        self.C_rcr_base = data['C_rcr_base']
        self.Rd_rcr_base = data['Rd_rcr_base']
        self.Rrv_base = data['Rrv_base']
        self.Rlv_base = data['Rlv_base']
        self.Rpd_base = data['Rpd_base']
        self.steps_per_cycle = data['steps_per_cycle']
        self.num_cycles = data['num_cycles']
        self.bc_vessel_map = data['bc_vessel_map']
    # ------------------------------------------------------------

    def get_data_members(self):
        data = {}
        data['names_corBC_l'] = self.names_corBC_l 
        data['names_corBC_r'] = self.names_corBC_r 
        data['names_RCRBC'] = self.names_RCRBC
        data['idxs_corBC_l'] = self.idxs_corBC_l 
        data['idxs_corBC_r'] = self.idxs_corBC_r 
        data['idxs_RCRBC'] = self.idxs_RCRBC
        data['n_corBC_l'] = self.n_corBC_l
        data['n_corBC_r'] = self.n_corBC_r
        data['n_corBC'] = self.n_corBC 
        data['n_RCRBC'] = self.n_RCRBC 
        data['Ra_l_base'] = self.Ra_l_base
        data['Ram_l_base'] = self.Ram_l_base 
        data['Rv_l_base'] = self.Rv_l_base
        data['Ca_l_base'] = self.Ca_l_base
        data['Cim_l_base'] = self.Cim_l_base
        data['Ra_r_base'] = self.Ra_r_base
        data['Ram_r_base'] = self.Ram_r_base
        data['Rv_r_base'] = self.Rv_r_base
        data['Ca_r_base'] = self.Ca_r_base
        data['Cim_r_base'] = self.Cim_r_base
        data['Rp_rcr_base'] = self.Rp_rcr_base
        data['C_rcr_base'] = self.C_rcr_base
        data['Rd_rcr_base'] = self.Rd_rcr_base
        data['Rrv_base'] = self.Rrv_base
        data['Rlv_base'] = self.Rlv_base
        data['Rpd_base'] = self.Rpd_base
        data['steps_per_cycle'] = self.steps_per_cycle
        data['num_cycles'] = self.num_cycles
        data['bc_vessel_map'] = self.bc_vessel_map
        data['p_conv'] = self.p_conv
        return data
    # ------------------------------------------------------------
    
    def parameter_names(self):
        names = ["Tsa", "tpwave", "Erv", "Elv", "iml", "Lrv_a", "Rrv_a", "Lra_v", "Rra_v", "Lla_v", 
                 "Rla_v", "Rlv_ao", "Llv_a", "Vrv_u", "Vlv_u", "Rpd", "Cp", "Cpa", "Kxp_ra", "Kxv_ra", 
                 "Emax_ra", "Vaso_ra", "Kxp_la", "Kxv_la", "Emax_la", "Vaso_la", "Ram_cor", "Rv_cor", 
                 "Cam_l", "Ca_l", "Cam_r", "Ca_r", "Rrcr", "Crcr", "imr"]
        return names
    # ------------------------------------------------------------
    
    def results_names(self):
        names = ["Pao-min", "Pao-min_conv", "Pao-max", "Pao-max_conv", "Pao-mean", "Pao-mean_conv", 
                 "Aor-Cor-split", "ABSQinlet", "ABSQinlet_conv", "Qsystole_perc", "Ppul-mean", 
                 "EF-LV", "ESV", "EDV", "Qla-ratio", "mit-valve-time", "aor-valve-time", 
                 "pul-valve-time", "Pra-mean", "l-cor-max-ratio", "l-cor-tot-ratio", "l-third-FF", 
                 "l-half-FF", "l-grad-ok", "r-cor-max-ratio", "r-cor-tot-ratio", "r-third-FF", 
                 "r-half-FF", "r-grad-ok"]
        return names
    # ------------------------------------------------------------
    
    def num_parameters(self):
        return 35
    # ------------------------------------------------------------
    
    def num_results(self):
        return 29
    # ------------------------------------------------------------
    
    def read_frozen_params(self, frozen_params_file):
        self.__frozen_param_idxs = np.genfromtxt(frozen_params_file, usecols=0, 
                                                 delimiter=',', dtype=int)
        self.__frozen_param_values = np.genfromtxt(frozen_params_file, usecols=1, 
                                                   delimiter=',', dtype=float)
        all_param_idxs = np.arange(self.num_parameters(), dtype=int)
        self.__variable_param_idxs = all_param_idxs[~np.in1d(all_param_idxs, self.__frozen_param_idxs)]

        return self.__frozen_param_idxs, self.__variable_param_idxs, self.__frozen_param_values
    
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
    
    def run_with_frozen_params(self, new_params):
        params = np.zeros(self.num_parameters())
        params[self.__frozen_param_idxs] = self.__frozen_param_values
        params[self.__variable_param_idxs]= new_params
        #print('new_params:', new_params)
        self.update_model_params(params)
        self.__solver.run()
        results = self.post_process() 
        #print('results: ', results)
        return results
    
    # ------------------------------------------------------------

    def evaluate_error(self, params):
        results = self.run_with_params(params)
        error = self.weighted_mse_error(results)
        return error
    
    # ------------------------------------------------------------

    def weighted_mse_error(self, results):
        scale = self.results_scale[self.targets_idxs]
        error = np.mean(np.divide(np.square((self.targets_values - results)[self.targets_idxs]), scale))
        print("Error = ", error)
        return error
    
    # ------------------------------------------------------------

    def update_model_params(self, params):
        coronary_params = np.zeros(5)
        for i, block_name in enumerate(self.names_corBC_l):
            coronary_params[0] = self.Ra_l_base[i]*params[26]
            coronary_params[1] = self.Ram_l_base[i]*params[26]
            coronary_params[2] = self.Rv_l_base[i]*params[27]
            coronary_params[3] = self.Ca_l_base[i]*params[29]
            coronary_params[4] = self.Cim_l_base[i]*params[28]
            self.__solver.update_block_params(block_name, coronary_params)
        
        for i, block_name in enumerate(self.names_corBC_r):
            coronary_params[0] = self.Ra_r_base[i]*params[26]
            coronary_params[1] = self.Ram_r_base[i]*params[26]
            coronary_params[2] = self.Rv_r_base[i]*params[27]
            coronary_params[3] = self.Ca_r_base[i]*params[31]
            coronary_params[4] = self.Cim_r_base[i]*params[30]
            self.__solver.update_block_params(block_name, coronary_params)
        
        rcr_params = np.zeros(3)
        for i, block_name in enumerate(self.names_RCRBC):
            rcr_params[0] = self.Rp_rcr_base[i]*params[32]
            rcr_params[1] = self.C_rcr_base[i]*params[33]
            rcr_params[2] = self.Rd_rcr_base[i]*params[32]
            self.__solver.update_block_params(block_name, rcr_params)
    	
        heart_params = np.zeros(27)
        heart_params[0]  = params[0]
        heart_params[1]  = params[1]
        #heart_params[2]  = params[2]
        heart_params[2]  = params[2]*self.p_conv #CGS
        #heart_params[3]  = params[3]
        heart_params[3]  = params[3]*self.p_conv #CGS
        heart_params[4]  = params[4]
        heart_params[5]  = params[34]
        #heart_params[6]  = params[7]/self.pConv; 
        heart_params[6]  = params[7] #CGS
        #heart_params[7]  = params[8]/self.pConv; 
        heart_params[7]  = params[8] #CGS
        #heart_params[8]  = params[5]/self.pConv; 
        heart_params[8]  = params[5] #CGS 
        #heart_params[9]  = self.Rrv_base*params[6]/self.pConv
        heart_params[9]  = self.Rrv_base*params[6] #CGS
        #heart_params[10] = params[9]/self.pConv
        heart_params[10] = params[9] #CGS
        #heart_params[11] = params[10]/self.pConv
        heart_params[11] = params[10] #CGS
        #heart_params[12] = params[12]/self.pConv
        heart_params[12] = params[12] #CGS
        #heart_params[13] = self.Rlv_base*params[11]/self.pConv
        heart_params[13] = self.Rlv_base*params[11] #CGS
        heart_params[14] = params[13]
        heart_params[15] = params[14]
        #heart_params[16] = self.Rpd_base*params[15]/self.pConv
        heart_params[16] = self.Rpd_base*params[15] #CGS
        #heart_params[17] = params[16]
        heart_params[17] = params[16]/self.p_conv #CGS
        #heart_params[18] = params[17]
        heart_params[18] = params[17]/self.p_conv #CGS
        #heart_params[19] = params[18]
        heart_params[19] = params[18]*self.p_conv #CGS
        heart_params[20] = params[19]
        #heart_params[21] = params[22]
        heart_params[21] = params[22]*self.p_conv #CGS
        heart_params[22] = params[23]
        #heart_params[23] = params[20]
        heart_params[23] = params[20]*self.p_conv #CGS
        #heart_params[24] = params[24]
        heart_params[24] = params[24]*self.p_conv #CGS
        heart_params[25] = params[21]
        heart_params[26] = params[25]
        self.__solver.update_block_params("CLH", heart_params) 
    
    # ------------------------------------------------------------

    def update_config_params(self, params):
        new_config = self.__config.copy()
        
        for i, bc_idx in enumerate(self.idxs_corBC_l):
            bc_values = new_config['boundary_conditions'][bc_idx]['bc_values'] 
            bc_values['Ra'] = self.Ra_l_base[i]*params[26]
            bc_values['Ram'] = self.Ram_l_base[i]*params[26]
            bc_values['Rv'] = self.Rv_l_base[i]*params[27]
            bc_values['Ca'] = self.Ca_l_base[i]*params[29]
            bc_values['Cim'] = self.Cim_l_base[i]*params[28]
        
        for i, bc_idx in enumerate(self.idxs_corBC_r):
            bc_values = new_config['boundary_conditions'][bc_idx]['bc_values'] 
            bc_values['Ra'] = self.Ra_r_base[i]*params[26]
            bc_values['Ram'] = self.Ram_r_base[i]*params[26]
            bc_values['Rv'] = self.Rv_r_base[i]*params[27]
            bc_values['Ca'] = self.Ca_r_base[i]*params[31]
            bc_values['Cim'] = self.Cim_r_base[i]*params[30]
        
        for i, bc_idx in enumerate(self.idxs_RCRBC):
            bc_values = new_config['boundary_conditions'][bc_idx]['bc_values'] 
            bc_values['Rp'] = self.Rp_rcr_base[i]*params[32]
            bc_values['C'] = self.C_rcr_base[i]*params[33]
            bc_values['Rd'] = self.Rd_rcr_base[i]*params[32]
    	
    	
        heart_params = new_config['closed_loop_blocks'][0]['parameters']
        heart_params['Tsa']  = params[0]
        heart_params['tpwave']  = params[1]
        #heart_params['Erv_s']  = params[2]
        heart_params['Erv_s']  = params[2]*self.p_conv #CGS
        #heart_params['Elv_s']  = params[3]
        heart_params['Elv_s']  = params[3]*self.p_conv #CGS
        heart_params['iml']  = params[4]
        heart_params['imr']  = params[34]
        #heart_params['Lra_v']  = params[7]/self.pConv; 
        heart_params['Lra_v']  = params[7] #CGS
        #heart_params['Rra_v']  = params[8]/self.pConv; 
        heart_params['Rra_v']  = params[8] #CGS 
        #heart_params['Lrv_a']  = params[5]/self.pConv; 
        heart_params['Lrv_a']  = params[5] #CGS 
        #heart_params['Rrv_a']  = self.Rrv_base*params[6]/self.pConv
        heart_params['Rrv_a']  = self.Rrv_base*params[6] #CGS
        #heart_params['Lla_v'] = params[9]/self.pConv
        heart_params['Lla_v'] = params[9] #CGS
        #heart_params['Rla_v'] = params[10]/self.pConv
        heart_params['Rla_v'] = params[10] #CGS
        #heart_params['Llv_a'] = params[12]/self.pConv
        heart_params['Llv_a'] = params[12] #CGS
        #heart_params['Rlv_ao'] = self.Rlv_base*params[11]/self.pConv
        heart_params['Rlv_ao'] = self.Rlv_base*params[11] #CGS
        heart_params['Vrv_u'] = params[13]
        heart_params['Vlv_u'] = params[14]
        #heart_params['Rpd'] = self.Rpd_base*params[15]/self.pConv
        heart_params['Rpd'] = self.Rpd_base*params[15] #CGS
        #heart_params['Cp'] = params[16]
        heart_params['Cp'] = params[16]/self.p_conv #CGS
        #heart_params['Cpa'] = params[17]
        heart_params['Cpa'] = params[17]/self.p_conv #CGS
        #heart_params['Kxp_ra'] = params[18]
        heart_params['Kxp_ra'] = params[18]*self.p_conv #CGS
        heart_params['Kxv_ra'] = params[19]
        #heart_params['Kxp_la'] = params[22]
        heart_params['Kxp_la'] = params[22]*self.p_conv #CGS
        heart_params['Kxv_la'] = params[23]
        #heart_params['Emax_ra'] = params[20]
        heart_params['Emax_ra'] = params[20]*self.p_conv #CGS
        #heart_params['Emax_la'] = params[24]
        heart_params['Emax_la'] = params[24]*self.p_conv #CGS
        heart_params['Vaso_ra'] = params[21]
        heart_params['Vaso_la'] = params[25]
        
        return new_config
    # ------------------------------------------------------------

    def read_params(self):
        coronary_params = np.zeros(5)
        for i, block_name in enumerate(self.names_corBC_l):
            coronary_params = self.__solver.read_block_params(block_name)
        
        for i, block_name in enumerate(self.names_corBC_r):
            coronary_params = self.__solver.read_block_params(block_name)
        
        rcr_params = np.zeros(3)
        for i, block_name in enumerate(self.names_RCRBC):
            rcr_params = self.__solver.read_block_params(block_name)

        heart_params = self.__solver.read_block_params("CLH")
    
    # ------------------------------------------------------------

    def get_result_weights(self):
        weights = np.zeros(self.num_results())
        weights[0] = 0.25  # PaoMin
        weights[1] = 0.25  # PaoMin_diff
        weights[2] = 0.25  # PaoMax
        weights[3] = 0.25  # PaoMax_diff
        weights[4] = 1.0   # PaoMean
        weights[5] = 0.25  # PaoMean_diff
        weights[6] = 1.0   # AorCorSplit
        weights[7] = 0.5   # AbsQin
        weights[8] = 0.25  # AbsQin_diff
        weights[9] = 1.0   # Qsystole_perc (maybe 999999.9 if rigid model?)
        weights[10] = 2.0  # PpulMean
        weights[11] = 1.0  # EFLV
        weights[12] = 0.5  # ESV
        weights[13] = 0.5  # EDV
        weights[14] = 2.0  # QlaRatio
        weights[15] = 2.0  # mitValveTime
        weights[16] = 2.0  # aorValveTime
        weights[17] = 2.0  # pulValveTime
        weights[18] = 1.0  # PraMean
        weights[19] = 0.5  # LCorMaxRatio
        weights[20] = 0.5  # LCorTotRatio
        weights[21] = 1.0  # LThirdFF
        weights[22] = 1.0  # LHalfFF
        weights[23] = 1.0  # LGradOK
        weights[24] = 0.5  # RCorMaxRatio
        weights[25] = 0.5  # RCorTotRatio
        weights[26] = 1.0  # RThirdFF
        weights[27] = 1.0  # RHalfFF
        weights[28] = 1.0  # RGradOK

        return weights
    
    # ------------------------------------------------------------
   
    def get_result_std(self):
        std = np.zeros(self.num_results())
        std[0] = 8.1     # PaoMin
        std[1] = 8.1     # PaoMin_diff
        std[2] = 12.6    # PaoMax
        std[3] = 12.6    # PaoMax_diff
        std[4] = 9.6     # PaoMean
        std[5] = 9.6     # PaoMean_diff
        std[6] = 0.4     # AorCorSplit
        std[7] = 9.07    # AbsQin
        std[8] = 9.07    # AbsQin_diff
        std[9] = 0.5     # Qsystole_perc
        std[10] = 3.3    # PpulMean
        std[11] = 0.065  # EFLV
        std[12] = 4.0    # ESV
        std[13] = 10.0   # EDV
        std[14] = 0.236  # QlaRatio
        std[15] = 0.084  # mitValveTime
        std[16] = 0.051  # aorValveTime
        std[17] = 0.051  # pulValveTime
        std[18] = 1.2    # PraMean
        std[19] = 0.8    # LCorMaxRatio
        std[20] = 2.5337 # LCorTotRatio
        std[21] = 0.02   # LThirdFF
        std[22] = 0.03   # LHalfFF
        std[23] = 1.00   # LGradOK
        std[24] = 0.3    # RCorMaxRatio
        std[25] = 1.0816 # RCorTotRatio
        std[26] = 0.07   # RThirdFF
        std[27] = 0.07   # RHalfFF
        std[28] = 1.00   # RGradOK

        # Convert pressures to CGS units
        std[0:6] *= self.p_conv
        std[10] *= self.p_conv
        std[18] *= self.p_conv

        return std

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
        limits[0]=0.39
        limits[1]=0.43 # Tsa
        limits[2]=8.43 
        limits[3]=9.31 # tpwave
        limits[4]=0.95
        limits[5]=3.33 # Erv
        limits[6]=1.14
        limits[7]=6.30 # Elv
        limits[8]=0.30
        limits[9]=0.88 # iml
        limits[10]=0.19
        limits[11]=0.7 # Lrv_a
        limits[12]=0.87
        limits[13]=1.83 # Rrv_a
        limits[14]=0.01
        limits[15]=0.84 # Lra_v
        limits[16]=7.77
        limits[17]=13.14 # Rra_v
        limits[18]=0.2
        limits[19]=1.2 # Lla_v
        limits[20]=4.78
        limits[21]=12.0 # Rla_v
        limits[22]=0.69
        limits[23]=1.63 # Rlv_ao
        limits[24]=0.1
        limits[25]=0.72 # Llv_a
        limits[26]=-10.0
        limits[27]=10.0 # Vrv_u
        limits[28]=-20.0
        limits[29]=5.0 # Vlv_u
        limits[30]=0.69
        limits[31]=1.80 # Rpd
        limits[32]=1.0
        limits[33]=1.15 # Cp
        limits[34]=0.05
        limits[35]=1.32 # Cpa
        limits[36]=1.0
        limits[37]=10.00 # Kxp_ra
        limits[38]=0.003
        limits[39]=0.0051 # Kxv_ra
        limits[40]=0.25
        limits[41]=0.50 # Emax_ra
        limits[42]=-5.00
        limits[43]=5.00 # Vaso_ra
        limits[44]=0.29
        limits[45]=10.73 # Kxp_la
        limits[46]=0.0078
        limits[47]=0.0085 # Kxv_la
        limits[48]=0.29
        limits[49]=0.32 # Emax_la
        limits[50]=-1.69
        limits[51]=15.81 # Vaso_la
        limits[52]=0.1
        limits[53]=1.5 # Ram_cor
        limits[54]=0.5
        limits[55]=10.0 # Rv_cor
        limits[56]=9.76
        limits[57]=17.96 # Cam_l
        limits[58]=1.84
        limits[59]=13.94 # Ca_l
        limits[60]=0.05
        limits[61]=15.28 # Cam_r
        limits[62]=0.46
        limits[63]=14.36 # Ca_r
        limits[64]=0.55
        limits[65]=1.69 # Rrcr
        limits[66]=0.1
        limits[67]=2.0 # Crcr
        limits[68]=0.2000
        limits[69]=1.28 # imr
        return limits
    # ------------------------------------------------------------
    
    def post_process(self):
        total_steps = self.num_cycles*(self.steps_per_cycle-1) + 1
        times = self.__solver.get_times()

        # Sum RCR flux
        Q_rcr = 0.0
        for i in range(self.n_RCRBC):
            var = self.__solver.get_single_result("flow:"+self.bc_vessel_map[self.names_RCRBC[i]]+":"+self.names_RCRBC[i])
            Q_rcr += np.trapezoid(var[-self.steps_per_cycle:], x=times[-self.steps_per_cycle:])
#       plt.figure()
#       plt.plot(times[-self.steps_per_cycle:],var[-self.steps_per_cycle:])
#       plt.savefig("Q_rcr.pdf")
#       plt.close()
         
        # Sum left coronary flux
        Q_lcor = 0.0
        for i in range(self.n_corBC_l):
            var = self.__solver.get_single_result("flow:"+self.bc_vessel_map[self.names_corBC_l[i]]+":"+self.names_corBC_l[i])
            Q_lcor += np.trapezoid(var[-self.steps_per_cycle:], x=times[-self.steps_per_cycle:])

        # Integrate left main flow
        q_lca_main = self.__solver.get_single_result("flow:"+self.bc_vessel_map[self.names_corBC_l[0]]+":"+self.names_corBC_l[0])
        lmain_flow = np.trapezoid(q_lca_main[-self.steps_per_cycle:], x=times[-self.steps_per_cycle:])
#       plt.figure()
#       plt.plot(times,q_lca_main)
#       plt.savefig("q_lca_main.pdf")
#       plt.close()

        # Sum right coronary flux
        Q_rcor = 0.0
        for i in range(self.n_corBC_r):
            var = self.__solver.get_single_result("flow:"+self.bc_vessel_map[self.names_corBC_r[i]]+":"+self.names_corBC_r[i])
            Q_rcor += np.trapezoid(var[-self.steps_per_cycle:], x=times[-self.steps_per_cycle:])

        # Integrate right main flow
        q_rca_main = self.__solver.get_single_result("flow:"+self.bc_vessel_map[self.names_corBC_r[0]]+":"+self.names_corBC_r[0])
        rmain_flow = np.trapezoid(q_rca_main[-self.steps_per_cycle:], x=times[-self.steps_per_cycle:])

        # End of systole
        q_lv = self.__solver.get_single_result("Q_LV:CLH")
#       plt.figure()
#       plt.plot(times,q_lv)
#       plt.savefig("q_lv.pdf")
#       plt.close()
        small_number = 1e-4
        systole_end = int(total_steps - self.steps_per_cycle/2) - 1
        for i in range(total_steps - self.steps_per_cycle - 1, total_steps-1):
            if(q_lv[i-1] > small_number and q_lv[i] > small_number and q_lv[i+1] < small_number):
                systole_end = i
                break

        # Start of systole
        systole_start = total_steps - self.steps_per_cycle - 1
        for i in range(total_steps - self.steps_per_cycle - 1, total_steps - 1):
            if(q_lv[i] < small_number and q_lv[i+1] > small_number and  q_lv[i+2] > small_number):
                systole_start = i
                break

        # Mitral valve opens
        q_la = self.__solver.get_single_result("Q_LA:CLH")
        mit_open = total_steps - self.steps_per_cycle - 1
        for i in range(total_steps - self.steps_per_cycle - 1, total_steps - 1):
            if(q_la[i] < small_number and q_la[i+1] > small_number and q_la[i+2] > small_number):
                mit_open = i
                break

        mit_half = int( round((mit_open + total_steps)/2.0) )
        aor_half = int( round((systole_start + systole_end)/2.0) )

        # Max and total coronary flow during systole
        l_cor_qmax_s = np.amax(q_lca_main[systole_start : systole_end])
        l_cor_qtot_s = np.trapezoid(q_lca_main[systole_start : systole_end], x=times[systole_start : systole_end])
        r_cor_qmax_s = np.amax(q_rca_main[systole_start : systole_end])
        r_cor_qtot_s = np.trapezoid(q_rca_main[systole_start : systole_end], x=times[systole_start : systole_end])

        # Max and total coronary flow during diastole
        sys_buffer = int(self.steps_per_cycle/10)
        l_cor_qmax_d = max( np.amax(q_lca_main[systole_end+sys_buffer : total_steps]), np.amax(q_lca_main[total_steps - self.steps_per_cycle - 1 : systole_start]) )
        l_cor_qtot_d = np.trapezoid(q_lca_main[systole_end : total_steps], x=times[systole_end : total_steps]) + np.trapezoid(q_lca_main[total_steps - self.steps_per_cycle - 1 : systole_start], x=times[total_steps - self.steps_per_cycle - 1 : systole_start])
        r_cor_qmax_d = max( np.amax(q_rca_main[systole_end+sys_buffer : total_steps]), np.amax(q_rca_main[total_steps - self.steps_per_cycle - 1 : systole_start]) )
        r_cor_qtot_d = np.trapezoid(q_rca_main[systole_end : total_steps], x=times[systole_end : total_steps]) + np.trapezoid(q_rca_main[total_steps - self.steps_per_cycle - 1 : systole_start], x=times[total_steps - self.steps_per_cycle - 1 : systole_start])

        # Ratios (Diastole to systole)
        l_cor_max_ratio = l_cor_qmax_d/l_cor_qmax_s
        l_cor_tot_ratio = l_cor_qtot_d/l_cor_qtot_s
        r_cor_max_ratio = r_cor_qmax_d/r_cor_qmax_s
        r_cor_tot_ratio = r_cor_qtot_d/r_cor_qtot_s

		# Find number of peaks and valleys in coronary flow waveforms
        l_grad_check = -1*np.ones(5)
        r_grad_check = -1*np.ones(5)
        l_last = (q_lca_main[total_steps-self.steps_per_cycle-1] - q_lca_main[total_steps-self.steps_per_cycle-2]) / (times[total_steps-self.steps_per_cycle-1] - times[total_steps-self.steps_per_cycle-2])
        r_last = (q_rca_main[total_steps-self.steps_per_cycle-1] - q_rca_main[total_steps-self.steps_per_cycle-2]) / (times[total_steps-self.steps_per_cycle-1] - times[total_steps-self.steps_per_cycle-2])
        for i in range(total_steps-self.steps_per_cycle, total_steps):
            l_grad = (q_lca_main[i] - q_lca_main[i-1])/(times[i] - times[i-1])
            r_grad = (q_rca_main[i] - q_rca_main[i-1])/(times[i] - times[i-1])

            # Checking the gradients on the left side
            if (l_grad > 0 and l_last <= 0 and l_grad_check[0] == -1): # valley
                l_grad_check[0] = i - (self.num_cycles - 1)*self.steps_per_cycle
                if(l_grad_check[1] < 0):
                    l_grad_check[4] = 1  # Starts with a valley?
            elif (l_grad < 0 and l_last >= 0 and l_grad_check[1] == -1): # peak
                l_grad_check[1] = i - (self.num_cycles - 1)*self.steps_per_cycle
            elif (l_grad > 0 and l_last <= 0 and l_grad_check[2] == -1 and l_grad_check[0] > 0): # valley
                l_grad_check[2] = i - (self.num_cycles - 1)*self.steps_per_cycle
            elif (l_grad < 0 and l_last >= 0 and l_grad_check[3] == -1 and l_grad_check[1] > 0 and (i - (self.num_cycles-1)*self.steps_per_cycle - l_grad_check[2]) > self.steps_per_cycle/10): # peak
                l_grad_check[3] = i - (self.num_cycles - 1)*self.steps_per_cycle
          
            # Check the gradients on the right side
            if (r_grad > 0 and r_last <= 0 and r_grad_check[0] == -1):
                r_grad_check[0] = i - (self.num_cycles-1)*self.steps_per_cycle
                if(r_grad_check[1] < 0):
                    r_grad_check[4] = 1
            elif(r_grad < 0 and r_last >= 0 and r_grad_check[1] == -1):
                r_grad_check[1] = i - (self.num_cycles-1)*self.steps_per_cycle
            elif(r_grad > 0 and r_last <= 0 and r_grad_check[2] == -1 and r_grad_check[0] > 0):
                r_grad_check[1] = i - (self.num_cycles-1)*self.steps_per_cycle
            elif(r_grad < 0 and r_last >= 0 and r_grad_check[3] == -1 and r_grad_check[1] > 0 and (i - (self.num_cycles-1)*self.steps_per_cycle - r_grad_check[2]) > self.steps_per_cycle/10):
                r_grad_check[3] = i - (self.num_cycles-1)*self.steps_per_cycle

            # Setting the last variables for next timestep
            l_last = l_grad;
            r_last = r_grad;

        # Tally up the good scores
        l_grad_ok = 0.0
        r_grad_ok = 0.0
        for i in range(5):
            if(l_grad_check[i] > 0):
                l_grad_ok = l_grad_ok + 1.0;
            if(r_grad_check[i] > 0):
                r_grad_ok = r_grad_ok + 1.0;

        # Calculate the 1/3 FF and 1/2 FF
        thirdCyc = int( round(self.steps_per_cycle/3) )
        halfCyc = int( round(self.steps_per_cycle/2) )
        if(systole_end+thirdCyc-1 < total_steps):
            r_third_FF = np.trapezoid( q_rca_main[systole_end-1 : systole_end+thirdCyc], x=times[systole_end-1 : systole_end+thirdCyc] )/rmain_flow
            l_third_FF = np.trapezoid( q_lca_main[systole_end-1 : systole_end+thirdCyc], x=times[systole_end-1 : systole_end+thirdCyc] )/lmain_flow
        else:
            r_third_FF = 0.0
            l_third_FF = 0.0

        if(systole_end+halfCyc-1 < total_steps):
            r_half_FF = np.trapezoid( q_rca_main[systole_end-1 : systole_end+halfCyc], x=times[systole_end-1 : systole_end+halfCyc] )/rmain_flow
            l_half_FF = np.trapezoid( q_lca_main[systole_end-1 : systole_end+halfCyc], x=times[systole_end-1 : systole_end+halfCyc] )/lmain_flow
        else:
            r_half_FF = 0.0
            l_half_FF = 0.0

        # Compute quantities
        q_aorta = self.__solver.get_single_result("flow:J_heart_outlet:aorta")
#       plt.figure()
#       plt.plot(times,q_aorta)
#       plt.savefig("q_aorta.pdf")
#       plt.close()
        Qinlet = np.trapezoid(q_aorta[total_steps - self.steps_per_cycle - 1 : total_steps], x=times[total_steps - self.steps_per_cycle - 1 : total_steps])
        Qsystole = np.trapezoid(q_aorta[systole_start:systole_end], x=times[systole_start:systole_end])
        systole_perc = Qsystole/Qinlet;
        Aor_Cor_split = ( (Q_lcor + Q_rcor) / (Q_lcor + Q_rcor + Q_rcr) ) * 100.0
        p_aorta = self.__solver.get_single_result("pressure:J_heart_outlet:aorta")
#       plt.figure()
#       plt.plot(times,p_aorta)
#       plt.savefig("p_aorta.pdf")
#       plt.close()
        Pao_max = np.amax(p_aorta[total_steps - self.steps_per_cycle - 1 : total_steps])
        Pao_min = np.amin(p_aorta[total_steps - self.steps_per_cycle - 1 : total_steps])
        Pao_mean = np.mean(p_aorta[total_steps - self.steps_per_cycle - 1 : total_steps])
        p_pul = self.__solver.get_single_result("P_pul:CLH") 
        Ppul_max = np.amax(p_pul[total_steps - self.steps_per_cycle - 1 : total_steps])
        Ppul_min = np.amin(p_pul[total_steps - self.steps_per_cycle - 1 : total_steps])
        Ppul_mean = np.mean(p_pul[total_steps - self.steps_per_cycle - 1 : total_steps])
        v_lv = self.__solver.get_single_result("V_LV:CLH") 
        ESV = np.amin(v_lv[total_steps - self.steps_per_cycle - 1 : total_steps])
        EDV = np.amax(v_lv[total_steps - self.steps_per_cycle - 1 : total_steps])
        EF_LV = (EDV - ESV)/EDV
        p_rv = self.__solver.get_single_result("P_RV:CLH")
        p_ra = self.__solver.get_single_result("pressure:J_heart_inlet:CLH")
        Prv_Pra = np.amax(p_rv[total_steps - self.steps_per_cycle - 1 : total_steps]) - np.amax(p_ra[total_steps - self.steps_per_cycle - 1 : total_steps])
        Ppul_Prv = np.amax(p_rv[total_steps - self.steps_per_cycle - 1 : total_steps]) - np.amax(p_pul[total_steps - self.steps_per_cycle - 1 : total_steps])
        mit_valve = np.zeros(self.steps_per_cycle) 
        aor_valve = np.zeros(self.steps_per_cycle)
        pul_valve = np.zeros(self.steps_per_cycle)
        step_ct = 0
        q_rv = self.__solver.get_single_result("Q_RV:CLH") 
        for step in range(total_steps-self.steps_per_cycle-1, total_steps-1):
            if(q_lv[step] > small_number and q_lv[step+1] > small_number):
                aor_valve[step_ct] = 1.0
            if(q_la[step] > small_number and q_la[step+1] > small_number):
                mit_valve[step_ct] = 1.0
            if(q_rv[step] > small_number and q_rv[step+1] > small_number):
                pul_valve[step_ct] = 1.0
            step_ct+=1
        mit_valve_time = float(np.sum(mit_valve)) / float(self.steps_per_cycle)
        aor_valve_time = float(np.sum(aor_valve)) / float(self.steps_per_cycle)
        pul_valve_time = float(np.sum(pul_valve)) / float(self.steps_per_cycle)
        Qla_ratio = np.amax(q_la[mit_open-1 : mit_half]) / np.amax(q_la[mit_half-1 : total_steps])
        Pra_mean = np.mean(p_ra[total_steps - self.steps_per_cycle - 1: total_steps])
        LR_split = (Q_lcor/(Q_lcor + Q_rcor))*100
        if(r_cor_max_ratio < 0 or l_cor_max_ratio < 0 or r_cor_tot_ratio < 0 or l_cor_tot_ratio < 0):
            r_cor_max_ratio = 9001.0
            l_cor_max_ratio = 9001.0
            r_cor_tot_ratio = 9001.0
            l_cor_tot_ratio = 9001.0

        # Compute convergence quantities
        Qinlet1 = np.trapezoid( q_aorta[total_steps - 2*self.steps_per_cycle - 1 : total_steps - self.steps_per_cycle], x=times[total_steps - self.steps_per_cycle - 1 : total_steps] )
        Qinlet_diff = abs(Qinlet - Qinlet1)
        Pao_max1 = np.amax(p_aorta[total_steps - 2*self.steps_per_cycle - 1 : total_steps - self.steps_per_cycle])
        Pao_max_diff = abs(Pao_max1 - Pao_max)
        Pao_min1 = np.amin(p_aorta[total_steps - 2*self.steps_per_cycle - 1 : total_steps - self.steps_per_cycle])
        Pao_min_diff = abs(Pao_min1 - Pao_min)
        Pao_mean1 = np.mean(p_aorta[total_steps - 2*self.steps_per_cycle - 1 : total_steps - self.steps_per_cycle])
        Pao_mean_diff = abs(Pao_mean1 - Pao_mean)

		# Save results
        results = np.zeros(29)
        results[0] = Pao_min
        results[1] = Pao_min_diff
        results[2] = Pao_max
        results[3] = Pao_max_diff
        results[4] = Pao_mean
        results[5] = Pao_mean_diff
        results[6] = Aor_Cor_split
        results[7] = abs(Qinlet)
        results[8] = Qinlet_diff
        results[9] = systole_perc
        results[10] = Ppul_mean
        results[11] = EF_LV
        results[12] = ESV
        results[13] = EDV
        results[14] = Qla_ratio
        results[15] = mit_valve_time
        results[16] = aor_valve_time
        results[17] = pul_valve_time
        results[18] = Pra_mean
        results[19] = l_cor_max_ratio
        results[20] = l_cor_tot_ratio
        results[21] = l_third_FF
        results[22] = l_half_FF
        results[23] = l_grad_ok
        results[24] = r_cor_max_ratio
        results[25] = r_cor_tot_ratio
        results[26] = r_third_FF
        results[27] = r_half_FF
        results[28] = r_grad_ok

        return results
    
    # ------------------------------------------------------------
    
    def read_targets_csv(self, targets_filename):
        target_names = np.genfromtxt(targets_filename, delimiter=',', usecols=0, dtype='U')
        target_values = np.genfromtxt(targets_filename, delimiter=',', usecols=2, dtype=float)
        if (len(target_names) != self.num_results()):
            raise RuntimeError('len(target_names) != self.num_results()')
        self.targets = dict(zip(target_names, target_values))
        self.targets_values = target_values
        self.targets_idxs = np.where(~np.isnan(self.targets_values))[0]
    
    # ------------------------------------------------------------
    
    def write_results_and_targets(self, results, savepath = '.'):
        
        write_data = np.array(list(zip(self.targets.keys(), self.targets.values(), results)), 
                              dtype=[('target_names','U16'),('target_values',float),('results',float)])
        
        savename = savepath + '/results_targets.dat'

        np.savetxt(savename, write_data, fmt = '%-16s %-18.10e %-18.10e', header='Name, Target, Result')
    
    # ------------------------------------------------------------
   
    def write_model_json(filename):
        with open(filename, 'w') as f:
            json.dump(json_0d, f, indent=4)
    
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

    data_folder = 'from_flow_mpi/afterClosedLoop/' 
    #f = open(data_folder+'svzerod_tuning.json')
    f = open(data_folder+'svzerod_tuning_cgs.json')
    original_config = json.load(f)
    
    model_dummy = Model_ClosedLoop(original_config)
    #model_dummy.read_targets_csv(data_folder+'coronary.csv')
    model_dummy.read_targets_csv(data_folder+'coronary_cgs.csv')
    params = np.genfromtxt(data_folder+'optParams.txt')
    model_dummy.update_model_params(params)
    results = model_dummy.run_model()
    model_dummy.write_results_and_targets(results)
    new_config = model_dummy.update_config_params(params)
    #with open(data_folder+'svzerod_updateconfig_closedloop.json', 'w') as f:
    with open(data_folder+'svzerod_updateconfig_closedloop_testcgs.json', 'w') as f:
            json.dump(new_config, f, indent=4)

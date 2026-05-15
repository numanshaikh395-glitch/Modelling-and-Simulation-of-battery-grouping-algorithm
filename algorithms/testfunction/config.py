# Central config — edit DATA_DIR before running

DATA_DIR  = r"E:\Thesis\Modelling_Simlulation_of optimization_code\Algorithms to be implemented\modified_data"
OUTPUT_DIR = r"E:\Thesis\Modelling_Simlulation_of optimization_code\Algorithms to be implemented\battery_sorting_results"

N_CELLS  = None   # None = load all files; set e.g. 500 to test on a subset
M        = 5      # cells per battery pack
WEIGHTS  = [0.5, 0.3, 0.2]   # [w_Q, w_R0, w_VOCV]

# Column names in your CSV files
COL_ID   = "cell_id"
COL_Q    = "capacity_Ah"
COL_R0   = "DCIR_est_mOhm"
COL_VOCV = "V_OCV_max_V"
COL_V_CURVE = "V_OCV"   # voltage curve column (timeseries)
COL_Q_STEP  = "q_step"  # step index for voltage curve

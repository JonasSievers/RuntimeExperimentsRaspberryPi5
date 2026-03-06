print("Start")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.New_Utils import main, make_cfg
import pandas as pd

print("Imported")

print("Local Learning Ausgrid Prosumption")
# Local Learning Ausgrid Prosumption
cfg = make_cfg(
    file_path="../../data/Final_Energy_Dataset_with_weather.csv.xz",
    columns="prosumption", nr_buildings=20, cluster_size=20, results_dir="../Results/LocalLearning",
    rounds=3,
    experiment_name=f"LL_Prosumption_Ausgrid_Allmodels",
    attack_enabled=False, use_federated=False, plot=False,
     models_to_run=["mlp", "cnn", "lstm", "transformer", "softdense", "softlstm"],
    plot_month_predictions=True,  use_weather=False, 
)
cfg["model_hyperparams"]["mlp"]["dense_units"] = 8
cfg["model_hyperparams"]["mlp"]["num_layers"] = 4
cfg["model_hyperparams"]["mlp"]["dropout"] = 0.2

combined_all, per_hour = main(cfg)

print("Done")

print("Local Learning Ausgrid PV")
# Local Learning Ausgrid PV
cfg = make_cfg(
    file_path="../../data/Final_Energy_Dataset_with_weather.csv.xz",
    columns="pv", nr_buildings=20, cluster_size=20, results_dir="../Results/LocalLearning",
    rounds=3,
    experiment_name=f"LL_PV_Ausgrid_Allmodels",
    attack_enabled=False, use_federated=False, plot=False,
     models_to_run=["mlp", "cnn", "lstm", "transformer", "softdense", "softlstm"],
    plot_month_predictions=True,  use_weather=False, 
)
cfg["model_hyperparams"]["mlp"]["dense_units"] = 8
cfg["model_hyperparams"]["mlp"]["num_layers"] = 4
cfg["model_hyperparams"]["mlp"]["dropout"] = 0.2

combined_all, per_hour = main(cfg)
print("Done")

print("Local Learning Ausgrid Load")
# Local Learning Ausgrid Load
cfg = make_cfg(
    file_path="../../data/Final_Energy_Dataset_with_weather.csv.xz",
    columns="load", nr_buildings=20, cluster_size=20, results_dir="../Results/LocalLearning",
    rounds=3,
    experiment_name=f"LL_Load_Ausgrid_Allmodels",
    attack_enabled=False, use_federated=False, plot=False,
     models_to_run=["mlp", "cnn", "lstm", "transformer", "softdense", "softlstm"],
    plot_month_predictions=True,  use_weather=False, 
)
cfg["model_hyperparams"]["mlp"]["dense_units"] = 8
cfg["model_hyperparams"]["mlp"]["num_layers"] = 4
cfg["model_hyperparams"]["mlp"]["dropout"] = 0.2

combined_all, per_hour = main(cfg)
print("Done")

print("Local Learning SPARK LOAD")
# Local Learning Ausgrid Load
cfg = make_cfg(
    file_path="../../data/Spark_Dataset_30m_2024_10loads.csv.xz",
    columns="load", nr_buildings=10, cluster_size=10, results_dir="../Results/LocalLearning",
    rounds=3,
    experiment_name=f"LL_Load_SPARK_Allmodels",
    attack_enabled=False, use_federated=False, plot=False,
     models_to_run=["mlp", "cnn", "lstm", "transformer", "softdense", "softlstm"],
    plot_month_predictions=True,  use_weather=False, 
)
cfg["model_hyperparams"]["mlp"]["dense_units"] = 8
cfg["model_hyperparams"]["mlp"]["num_layers"] = 4
cfg["model_hyperparams"]["mlp"]["dropout"] = 0.2

combined_all, per_hour = main(cfg)
print("Done")
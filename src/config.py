import json
import os
from pathlib import Path

#dataset = "DIRG"
dataset = "MAFAULDA"
TASK = 7

ROOT_DIR = Path(__file__).parent.parent
DATA_ROOT = ROOT_DIR / "data"
DIRG_DATA_DIR = DATA_ROOT / dataset
OUTPUT_ROOT = ROOT_DIR
LOGS_DIR = OUTPUT_ROOT / "logs"
MODELS_DIR = OUTPUT_ROOT / "models"
FIGURES_DIR = OUTPUT_ROOT

device = "cuda"
num_workers = 0
pin_memory = True
persistent_workers = True
prefetch_factor = 2
#DIRG任务划分
DIRG_TASK_DOMAINS = {
    1: {
        'src': [(100,0),(200,0),(300,0),(400,0),(100,700),(200,700),(300,700),(400,700)],  
        'tgt': [(100,500),(200,500),(300,500),(400,500)]   
    },
    2: {
        'src': [(100,0),(100,500),(100,700),(100,900),(300,0),(300,500),(300,700),(300,900)], 
        'tgt': [(200,0),(200,500),(200,700),(200,900)] 
    },
    3: {
        'src': [(100,0),(200,0),(300,0),(400,0),(100,500),(200,500),(300,500),(400,500)], 
        'tgt': [(100,700),(200,700),(300,700),(400,700)] 
    },
    4: {
        'src': [(100,0),(100,700),(300,0),(300,700)], 
        'tgt': [(200,0),(400,0),(100,500),(300,500),(200,700),(400,700),(100,900),(300,900)] 
    },
    5:{
        'src': [(25,0),(25,6),(25,20),(35,0),(35,6),(35,20)], 
        'tgt': [(15,0),(15,6),(15,20),(30,0),(30,6),(30,20)]
    },
    6:{
        'src': [(15,0),(25,0),(30,0),(35,0),(45,0),(15,20),(25,20),(30,20),(35,20)], 
        'tgt': [(15,6),(25,6),(30,6),(35,6),(45,6)]
    },
    7:{
        'src': [(30,0),(30,6),(30,20)], 
        'tgt': [(15,0),(15,6),(15,20),(25,0),(25,6),(25,20),(45,0),(45,6)]
    },
    8:{
        'src': [(15,6),(25,6),(30,6),(35,6),(45,6)], 
        'tgt': [(15,0),(15,20),(25,0),(25,20),(30,0),(20,20),(35,0),(35,20),(45,0)]
    }
}
DIRG_task_src = DIRG_TASK_DOMAINS[TASK]['src']
DIRG_task_tgt = DIRG_TASK_DOMAINS[TASK]['tgt']

# MCFD-ML weights. Legacy scripts may still refer to the implementation as MEDG.
num_classes=7
epochs = 100
channels = 8
weight_outer = 0.5
weight_coral=0.3
weight_adv = 1.0
weight_domainacc = 0.2
weight_HSIC = 0.1
weight_rec = 0.2
batch_size = 64
lr = 0.0001
lr_decay_enabled = False
lr_decay_step_size = 20
lr_decay_gamma = 0.9
medg_ablation = "none"
medg_method_name = "MCFD-ML"

#DANN0权重
DANN0_num_classes = 7
DANN0_epochs = 100
DANN0_weight_domain = 0.5
DANN0_batch_size = 128
DANN0_lr = 0.0005

#DANN权重
DANN_num_classes = 7
DANN_epochs = 100
DANN_weight_domain = 1
DANN_batch_size = 128
DANN_lr = 0.0005

#MCD权重
MCD_num_classes = 7
MCD_epochs = 100
MCD_batch_size = 128
MCD_lr = 0.0005

#CDAN权重
CDAN_num_classes = 7
CDAN_epochs = 100
CDAN_lr = 0.0005
CDAN_batch_size = 64
CDAN_entropy = True
CDAN_trade_off = 1.5


#域分析
pretrained_model_path = MODELS_DIR / "task7_43_98.92.pt"
Domain_num_classes = 7
domain_num = 11

#ERM
ERM_num_classes = 7
ERM_epochs = 100
ERM_batch_size = 128
ERM_lr = 0.0005

#MLDG
MLDG_inner_lr = 0.001
MLDG_beta = 1.0

#CDDG
CDDG_epochs = 100
CDDG_batch_size = 64
CDDG_lr = 0.0001


def _apply_runtime_config():
    runtime_config_path = os.environ.get("MCED_RUNTIME_CONFIG")
    if not runtime_config_path:
        return
    with open(runtime_config_path, "r", encoding="utf-8-sig") as f:
        overrides = json.load(f)
    for key, value in overrides.items():
        if key.startswith("_"):
            continue
        if key.endswith("_DIR") or key.endswith("_ROOT") or key in {"DIRG_DATA_DIR"}:
            value = Path(value)
            if not value.is_absolute():
                value = ROOT_DIR / value
        globals()[key] = value

    if "DIRG_DATA_DIR" not in overrides:
        globals()["DIRG_DATA_DIR"] = DATA_ROOT / dataset
    globals()["LOGS_DIR"] = Path(LOGS_DIR)
    globals()["MODELS_DIR"] = Path(MODELS_DIR)
    globals()["FIGURES_DIR"] = Path(FIGURES_DIR)
    globals()["LOGS_DIR"].mkdir(parents=True, exist_ok=True)
    globals()["MODELS_DIR"].mkdir(parents=True, exist_ok=True)
    globals()["FIGURES_DIR"].mkdir(parents=True, exist_ok=True)
    globals()["DIRG_task_src"] = DIRG_TASK_DOMAINS[TASK]["src"]
    globals()["DIRG_task_tgt"] = DIRG_TASK_DOMAINS[TASK]["tgt"]


_apply_runtime_config()

for _output_dir in (LOGS_DIR, MODELS_DIR, FIGURES_DIR):
    Path(_output_dir).mkdir(parents=True, exist_ok=True)

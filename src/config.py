from pathlib import Path

#dataset = "DIRG"
dataset = "MAFAULDA"
TASK = 7

ROOT_DIR = Path(__file__).parent.parent
DIRG_DATA_DIR = ROOT_DIR / "data" / dataset
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"

device = "cuda"
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

#MEDG权重
num_classes=7
epochs = 100
channels = 8
weight_outer = 0.5
weight_coral=0.3
weight_adv = 1
weight_domainacc = 0.2
weight_HSIC = 0.1
weight_rec = 0.2
batch_size = 128
lr = 0.0005

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
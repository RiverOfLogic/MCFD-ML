# 鑷姩鍖栬缁冭繍琛岃鏄?
鏈」鐩彁渚?`scripts/run_experiments.py`锛岀敤浜庢壒閲忚繍琛屽鏂规硶銆佸浠诲姟銆佸闅忔満绉嶅瓙鐨勫疄楠岋紝骞惰嚜鍔ㄦ眹鎬荤粨鏋溿€?
## 1. 閰嶇疆鏂囦欢

榛樿閰嶇疆鏂囦欢锛?
```powershell
experiments\auto_train.yaml
```

榛樿瀹為獙瑙勬ā锛?
- 鏂规硶锛歚ERM`, `DANN`, `M-DANN`, `CDAN`, `MCD`, `MLDG`, `CDDG`, `MCFD-ML`
- 浠诲姟锛歚1-8`
- 姣忎釜浠诲姟閲嶅锛歚10` 娆?- 榛樿 seeds锛歚42-51`
- task `1-4` 浣跨敤 `DIRG`锛岃緭鍏ラ€氶亾鏁?`6`
- task `5-8` 浣跨敤 `MAFAULDA`锛岃緭鍏ラ€氶亾鏁?`8`

甯哥敤鍙敼椤癸細

```yaml
methods: [MCFD-ML, CDAN]
tasks: [1, 2, 5, 6]
repeats: 10
base_seed: 42
gpus: auto
max_jobs_per_gpu: auto
output_dir: experiments/results/{timestamp}
data_root: data
```

濡傛灉鍙兂鎸囧畾閮ㄥ垎 GPU锛?
```yaml
gpus: [0, 1]
max_jobs_per_gpu: auto
```

## 2. GPU 鑷姩骞跺彂

榛樿閰嶇疆浣跨敤鏄惧瓨鎰熺煡璋冨害锛?
```yaml
gpus: auto
max_jobs_per_gpu: auto

gpu_scheduler:
  mode: memory
  reserve_mb: 1024
  min_free_mb: 512
  poll_interval_sec: 5
  startup_grace_sec: 30
  default_job_memory_mb: 3000
  max_jobs_per_gpu_cap: 4
  method_memory_mb:
    ERM: 2200
    DANN: 2600
    M-DANN: 3000
    CDAN: 3400
    MCD: 3000
    MLDG: 3800
    CDDG: 3800
    MCFD-ML: 4200
```

鍚箟锛?
- `max_jobs_per_gpu: auto`锛氫笉鍐嶅浐瀹氭瘡寮犲崱璺戝嚑涓繘绋嬶紝鐢辩┖闂叉樉瀛樺姩鎬佸喅瀹氥€?- `reserve_mb`锛氭瘡寮?GPU 棰勭暀鐨勬樉瀛橈紝閬垮厤鎶婃樉瀛樺畬鍏ㄥ帇婊°€?- `min_free_mb`锛氬惎鍔ㄦ柊浠诲姟鍚庝粛甯屾湜淇濈暀鐨勬渶浣庣┖闂叉樉瀛樸€?- `method_memory_mb`锛氭瘡绉嶆柟娉曚竴娆¤缁冨ぇ绾﹂渶瑕佺殑鏄惧瓨锛岀敤浜庤皟搴︿及绠椼€?- `max_jobs_per_gpu_cap`锛氬崟寮?GPU 鏈€澶氬苟鍙戝灏戜釜杩涚▼锛岄槻姝㈣繃搴﹀爢鍙犮€?- `startup_grace_sec`锛氭柊杩涚▼鍒氬惎鍔ㄦ椂锛宍nvidia-smi` 鍙兘杩樻病鏄剧ず鐪熷疄鍗犵敤锛岃皟搴﹀櫒浼氫复鏃舵寜棰勪及鏄惧瓨鎵ｉ櫎銆?
濡傛灉鎯虫洿婵€杩涘湴鍘嬫弧 GPU锛屽彲浠ラ€愭璋冨皬锛?
```yaml
reserve_mb: 512
min_free_mb: 256
```

鎴栬€呰皟灏忔煇涓柟娉曠殑浼拌鏄惧瓨锛屼緥濡傦細

```yaml
method_memory_mb:
  MCFD-ML: 3600
```

濡傛灉鍑虹幇 OOM锛屾妸瀵瑰簲鏂规硶鐨?`method_memory_mb` 璋冨ぇ锛屾垨鑰呮妸 `max_jobs_per_gpu_cap` 璋冨皬銆?
濡傛灉鎯冲洖鍒板浐瀹氬苟鍙戯細

```yaml
max_jobs_per_gpu: 1
```

鎴栵細

```yaml
max_jobs_per_gpu: 2
```

姝ゆ椂璋冨害鍣ㄤ細鎸夊浐瀹?slot 杩愯锛屼笉浣跨敤鏄惧瓨鍔ㄦ€佽皟搴︺€?
## 3. Dry-run 妫€鏌?
姝ｅ紡杩愯鍓嶅缓璁厛 dry-run锛?
```powershell
python scripts\run_experiments.py --config experiments\auto_train.yaml --dry-run
```

瀹冧細鎵撳嵃锛?
- 鎬?job 鏁伴噺
- 妫€娴嬪埌鐨?GPU
- 璋冨害妯″紡
- 褰撳墠 GPU 绌洪棽鏄惧瓨
- task 鍒?dataset/channel 鐨勬槧灏?- 鍓?20 涓缁冧换鍔?- 姣忎釜浠诲姟鐨勪及璁℃樉瀛?
榛樿搴旂湅鍒帮細

```text
jobs=640
DIRG/channels=6
MAFAULDA/channels=8
```

## 4. 姝ｅ紡杩愯

```powershell
python scripts/run_experiments.py --config experiments/auto_train.yaml
```

姣忎釜瀛愪换鍔′細鍗曠嫭鍚姩涓€涓?Python 杩涚▼锛屽苟璁剧疆锛?
- `CUDA_VISIBLE_DEVICES`
- `MCED_RUNTIME_CONFIG`
- `LOKY_MAX_CPU_COUNT=1`

涓嶄細骞跺彂鏀瑰啓 `src/config.py`銆?
## 5. 鏂偣缁窇

濡傛灉涓€斿仠姝紝鍙互鐢細

```powershell
python scripts\run_experiments.py --config experiments\auto_train.yaml --resume
```

鑴氭湰浼氳鍙栧凡鏈?`raw_runs.csv`锛岃烦杩囧凡缁?`success` 鐨?`(method, task, seed)`銆?
## 6. 灏忚妯℃祴璇?
绗竴娆″缓璁厛鎶?YAML 鏀瑰皬锛屼緥濡傦細

```yaml
methods: [MCFD-ML]
tasks: [1]
repeats: 2
max_jobs_per_gpu: auto
```

骞朵复鏃舵妸瀵瑰簲鏂规硶鐨?`epochs` 鏀规垚 `1`锛岀‘璁ゆ祦绋嬨€佹棩蹇楀拰 CSV 閮芥甯稿悗锛屽啀鎭㈠瀹屾暣瀹為獙銆?
## 7. 杈撳嚭鐩綍

榛樿杈撳嚭鍒帮細

```text
experiments/results/{timestamp}/
```

涓昏鏂囦欢锛?
```text
raw_runs.csv
summary.csv
runs/{method}/task{n}/seed{s}.log
runtime_configs/{method}_task{n}_seed{s}.json
logs/{method}_training.log
models/*.pt
figures/*.pdf
```

鑷姩鍖栬剼鏈細鎶婃瘡涓瓙杩涚▼鐨勮繍琛屾椂閰嶇疆鍐欏叆 `runtime_configs/`锛屽苟鎶婅缁冭剼鏈唴閮ㄤ娇鐢ㄧ殑鐩綍瑕嗙洊涓猴細

- `config.LOGS_DIR` -> `{output_dir}/logs`
- `config.MODELS_DIR` -> `{output_dir}/models`
- `config.FIGURES_DIR` -> `{output_dir}/figures`

鍥犳閫氳繃 `scripts/run_experiments.py` 鍚姩鏃讹紝CSV銆佸瓙杩涚▼鏃ュ織銆佽缁冩棩蹇椼€佹ā鍨?checkpoint銆丮EDG/MLDG 鐨?t-SNE 鍜屾贩娣嗙煩闃?PDF 閮戒細杩涘叆鏈瀹為獙鐨?`output_dir`銆傚崟鐙繍琛屾煇涓?`src/*.py` 鏃朵粛浣跨敤椤圭洰鏍圭洰褰曚笅鐨勯粯璁?`logs/`銆乣models/` 绛夌洰褰曘€?
`raw_runs.csv` 鏄瘡涓€娆¤缁冪殑缁撴灉锛屽寘鍚細

- method
- dataset
- task
- repeat
- seed
- gpu
- status
- acc
- macro_f1
- weighted_f1
- loss
- log_path
- model_path
- duration_sec

`summary.csv` 鏄寜 `method + dataset + task` 鑱氬悎鍚庣殑鍧囧€煎拰鏍囧噯宸紝鍖呭惈锛?
- acc_mean / acc_std
- weighted_f1_mean / weighted_f1_std
- macro_f1_mean / macro_f1_std
- loss_mean / loss_std

## 8. 璋冨弬浣嶇疆

鎵€鏈夋壒閲忓疄楠屽弬鏁颁紭鍏堝湪 YAML 涓敼锛屼笉寤鸿涓轰簡鎵归噺瀹為獙鐩存帴鏀?`src/config.py`銆?
渚嬪 MCFD-ML 鍦ㄤ袱涓暟鎹泦涓婂彲浠ュ垎鍒厤缃細

```yaml
params:
  MCFD-ML:
    DIRG:
      epochs: 100
      batch_size: 64
      lr: 0.0001
      weight_outer: 0.5
      weight_coral: 0.3
      weight_adv: 1.0
      weight_domainacc: 0.2
      weight_HSIC: 0.1
      weight_rec: 0.2
    MAFAULDA:
      epochs: 100
      batch_size: 64
      lr: 0.0001
      weight_outer: 0.5
      weight_coral: 0.3
      weight_adv: 1.0
      weight_domainacc: 0.2
      weight_HSIC: 0.1
      weight_rec: 0.2
```

濡傛灉鏌愪釜鏂规硶鎴栨煇涓暟鎹泦鏄惧瓨鍗犵敤鏄庢樉涓嶅悓锛屼篃鍙互鍦ㄥ搴斿弬鏁伴噷鍗曠嫭瑕嗙洊锛?
```yaml
params:
  MCFD-ML:
    DIRG:
      gpu_memory_mb: 3800
    MAFAULDA:
      gpu_memory_mb: 4600
```

## 9. 鏁版嵁璺緞

鏁版嵁鐩綍涔熷彲浠ュ湪 YAML 閲屾寚瀹氥€傞粯璁ゅ啓娉曪細

```yaml
data_root: data

datasets:
  DIRG:
    tasks: [1, 2, 3, 4]
    channels: 6
    path: data/DIRG
  MAFAULDA:
    tasks: [5, 6, 7, 8]
    channels: 8
    path: data/MAFAULDA
```

`path` 鍙互鏄浉瀵硅矾寰勬垨缁濆璺緞銆傜浉瀵硅矾寰勪細鎸夐」鐩牴鐩綍瑙ｆ瀽銆傛瘡涓换鍔″惎鍔ㄥ墠锛岃皟搴﹀櫒浼氭妸褰撳墠鏁版嵁闆嗙殑 `path` 鍐欏叆 runtime config锛屽苟瑕嗙洊 `config.DIRG_DATA_DIR`锛屾墍浠ユ棫璁粌鑴氭湰浠嶇劧閫氳繃 `config.DIRG_DATA_DIR / "train_x.npy"` 璇诲彇鏁版嵁銆?
`num_workers` 涔熷缓璁斁鍦?YAML 涓帶鍒讹細

```yaml
defaults:
  num_workers: 2
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
```

Linux 鏈嶅姟鍣ㄤ笂鎺ㄨ崘鍏堢敤 `num_workers=2`銆傚鏋?GPU 鍒╃敤鐜囦粛鐒朵笂涓嶅幓锛屽彲浠ヨ瘯 `4`锛涘鏋?CPU 鎴栧唴瀛樺帇鍔涘緢澶э紝鎴栬€?DataLoader worker 鎶ラ敊锛屽氨闄嶅洖 `1` 鎴?`0`銆?
瀹為檯鎬?worker 鏁板ぇ绾︽槸锛?
```text
鍚屾椂璁粌杩涚▼鏁?脳 num_workers
```

渚嬪 4 涓缁冭繘绋嬨€乣num_workers=2`锛屽ぇ绾︿細棰濆寮€ 8 涓?DataLoader worker銆備笉瑕佸湪楂樺苟鍙戜笅鐩存帴鎶?`num_workers` 璁惧埌 `8`銆?
## 10. 娉ㄦ剰浜嬮」

- 鑷姩骞跺彂渚濊禆 `nvidia-smi`銆傚鏋滄煡璇㈠け璐ワ紝鑴氭湰浼氬洖閫€鍒板浐瀹氬苟鍙戞ā寮忋€?- 璋冨害鍣ㄥ惎鍔ㄥ墠浼氭鏌ユ瘡涓暟鎹泦鐩綍鏄惁鍖呭惈 `train_x.npy`銆乣train_y.npy`銆乣train_info.npy`銆乣val_*`銆乣test_*`銆?- 濡傛灉鏌愪釜 job 澶辫触锛屼富缁堢浼氭墦鍗拌 job 鏃ュ織鏈€鍚庤嫢骞茶锛屼紭鍏堢湅杩欓噷鐨?traceback銆?- `DANN` 瀵瑰簲鏍囧噯浜屽煙 DANN锛屽叆鍙ｆ槸 `src/DANN0.py`銆?- `M-DANN` 瀵瑰簲澶氬煙 DANN锛屽叆鍙ｆ槸 `src/DANN.py`銆?- 鍗曠嫭杩愯鏃ц缁冭剼鏈粛鐒跺彲鐢紱鍙湁鑷姩鍖栬剼鏈細璁剧疆 `MCED_RUNTIME_CONFIG` 瑕嗙洊閰嶇疆銆?

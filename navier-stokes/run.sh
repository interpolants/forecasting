ROOT_DIR="/gpfs/scratch/goldsm20/forecasting/nvidia"
cd ${ROOT_DIR}
nvidia-smi
BETA='t^2' # 't' 
SIGMA=10.0 
py="python main.py --beta_fn ${BETA} --sigma_coef ${SIGMA} --use_wandb 1 --debug 0 --overfit 1"
eval ${py}

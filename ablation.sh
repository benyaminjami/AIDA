sbatch slurm.sh tags="focalloss" task.criterion._target_=src.utils.criterion.FocalLoss

sbatch slurm.sh tags="without_paired_pt" model.cfg.decoder.pretrained_path="/h/benjami/scrach/Conditional-BALM/data/pretrained_models/pretrained-BALM"

sbatch slurm.sh tags="selected_mask_cdr" task.learning.noise="selected_mask" task.learning.mask="cdr_weights" train.monitor="val/cdr_acc"

sbatch slurm.sh tags="selected_mask_h3" task.learning.noise="selected_mask" task.learning.mask="h3" train.monitor="val/h3_acc"

sbatch slurm.sh tags="selected_mask_full" task.learning.noise="full_mask" train.monitor="val/full_acc"

sbatch slurm.sh tags="antigen_adapter" model.cfg.decoder.balm_config.adapter_layer_indices=[]
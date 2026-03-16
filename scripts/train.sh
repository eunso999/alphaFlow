# MeanFlow
# ------------------------------------------------------------------------------------------------------------------------------
python infra/launch.py alphaflow-meanflow-latentspace-B-2 wandb.enabled=true 2>&1 | tee ./logs/log_meanflow_b2.txt
python infra/launch.py alphaflow-meanflow-latentspace-B-2-cfg-training wandb.enabled=true 2>&1 | tee ./logs/log_meanflow_b2_cfg.txt
python infra/launch.py alphaflow-meanflow-latentspace-XL-2-cfg-training wandb.enabled=true 2>&1 | tee ./logs/log_meanflow_xl2_cfg.txt

# analysis exp1
python infra/launch.py alphaflow-flowmatching-latentspace-B-2 wandb.enabled=true 2>&1 | tee ./logs/log_flowmatch_b2.txt

# analysis exp3-(1)
python infra/launch.py alphaflow-meanflow-latentspace-B-2-uniform_sampling wandb.enabled=true 2>&1 | tee ./logs/log_meanflow_b2_uniform_sampling.txt
python infra/launch.py alphaflow-meanflow-latentspace-B-2-uniform_sampling-analysis wandb.enabled=true 2>&1 | tee ./logs/log_meanflow_b2_uniform_sampling_analysis.txt
python infra/launch.py alphaflow-meanflow-latentspace-B-2-uniform_sampling-pretrained-analysis wandb.enabled=true 2>&1 | tee ./logs/log_meanflow_b2_uniform_sampling_analysis.txt

# analysis exp3-(2)
python infra/launch.py alphaflow-meanflow-latentspace-B-2-incremental_uniform_sampling wandb.enabled=true 2>&1 | tee ./logs/log_meanflow_b2_incremental_uniform_sampling.txt
python infra/launch.py alphaflow-meanflow-latentspace-B-2-incremental_uniform_sampling-analysis wandb.enabled=true 2>&1 | tee ./logs/log_meanflow_b2_incremental_uniform_sampling_analysis.txt
python infra/launch.py alphaflow-meanflow-latentspace-B-2-incremental_uniform_sampling-pretrained-analysis wandb.enabled=true 2>&1 | tee ./logs/log_meanflow_b2_incremental_uniform_sampling_analysis.txt

# additional analysis exp
python infra/launch.py alphaflow-meanflow-latentspace-B-2-uniform_sampling-additional_analysis_1 wandb.enabled=true 2>&1 | tee ./logs/log_meanflow_b2_uniform_sampling_additional_analysis_1.txt


# alphaflow
# ------------------------------------------------------------------------------------------------------------------------------
python infra/launch.py alphaflow-sigmoid-latentspace-B-2 wandb.enabled=true 2>&1 | tee ./logs/log_alphaflow_b2.txt
python infra/launch.py alphaflow-sigmoid-latentspace-B-2-cfg-training wandb.enabled=true 2>&1 | tee ./logs/log_alphaflow_b2_cfg.txt


# i-MeanFlow
# ------------------------------------------------------------------------------------------------------------------------------
python infra/launch.py alphaflow-improved-meanflow-latentspace-B-2 wandb.enabled=true 2>&1 | tee ./logs/log_imeanflow_b2.txt
python infra/launch.py alphaflow-improved-meanflow-latentspace-B-2-uniform_sampling-analysis wandb.enabled=true 2>&1 | tee ./logs/log_imeanflow_b2_uniform_sampling_analysis.txt

# additional analysis exp
python infra/launch.py alphaflow-improved-meanflow-latentspace-B-2-uniform_sampling-additional_analysis_1 wandb.enabled=true 2>&1 | tee ./logs/log_imeanflow_b2_uniform_sampling_additional_analysis_1.txt
python infra/launch.py alphaflow-improved-meanflow-latentspace-B-2-uniform_sampling-additional_analysis_2 wandb.enabled=true 2>&1 | tee ./logs/log_imeanflow_b2_uniform_sampling_additional_analysis_2.txt
python infra/launch.py alphaflow-improved-meanflow-latentspace-B-2-uniform_sampling-additional_analysis_3 wandb.enabled=true 2>&1 | tee ./logs/log_imeanflow_b2_uniform_sampling_additional_analysis_3.txt


# MeanFlow + CAC
# ------------------------------------------------------------------------------------------------------------------------------
python infra/launch.py alphaflow-meanflow-latentspace-B-2-uniform_sampling-iC2 wandb.enabled=true 2>&1 | tee ./logs/log_meanflow_b2_uniform_sampling_ic2.txt
python infra/launch.py alphaflow-meanflow-latentspace-B-2-uniform_sampling-iC2-analysis1 wandb.enabled=true 2>&1 | tee ./logs/log_meanflow_b2_uniform_sampling_ic2_analysis1.txt

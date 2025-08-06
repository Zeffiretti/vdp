#!/bin/bash

# Check if correct number of arguments is provided
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --global_obs) global_obs="$2"; shift ;;
        --global_action) global_action="$2"; shift ;;
        --global_horizon) global_horizon="$2"; shift ;;
        --config_choice) config_choice="$2"; shift ;;
        --past_action_pred) past_action_pred="$2"; shift ;;
        --past_steps_reg) past_steps_reg="$2"; shift ;;
        --vjepa2) vjepa2="$2"; shift ;;
        --emb) emb="$2"; shift ;;
        --cached) cached="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 --global_obs N --global_action N --global_horizon N --config_choice NAME --past_action_pred BOOL --past_steps_reg N --vjepa2 BOOL --emb BOOL --cached BOOL"
            echo "config_choice: [tool | square | square_past | transport | aloha | ...]"
            exit 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1 ;;
    esac
    shift
done

# Validate required arguments
if [[ -z "$global_obs" || -z "$global_action" || -z "$global_horizon" || -z "$config_choice" || -z "$past_action_pred" || -z "$past_steps_reg" || -z "$emb" || -z "$cached" ]]; then
    echo "Missing required arguments."
    echo "Run with -h or --help to see usage."
    exit 1
fi

# Validate config_choice and set CONFIG
case $config_choice in
  tool)
    CONFIG="transformer_tool_hang"
	CONFIG_DIR="experiment_configs/tool"
    ;;
  square)
	CONFIG="transformer_square"
	CONFIG_DIR="experiment_configs/square"
	;;
  square_past)
	CONFIG="transformer_square_past"
	CONFIG_DIR="experiment_configs/square"
	;;
  transport)
	CONFIG="transformer_transport"
	CONFIG_DIR="experiment_configs/transport"
	;;
  aloha)
	CONFIG="transformer_aloha"
	CONFIG_DIR="experiment_configs/aloha"
	;;
  aloha_real)
	CONFIG="transformer_aloha_real"
	CONFIG_DIR="experiment_configs/real"
	;;
  aloha_replace)
	CONFIG="transformer_aloha_replace"
	CONFIG_DIR="experiment_configs/real"
	;;
  franka_unload)
	CONFIG="transformer_franka_unload"
	CONFIG_DIR="experiment_configs/real"
	;;
  franka_unload_dct)
	CONFIG="transformer_franka_unload_dct"
	CONFIG_DIR="experiment_configs/real"
	;;
  longhist)
	CONFIG="transformer_longhist"
	CONFIG_DIR="experiment_configs/longhist"
	;;
  longhist_past)
	CONFIG="transformer_longhist_past"
	CONFIG_DIR="experiment_configs/longhist"
	;;
  video)
	CONFIG="transformer_vide"
	CONFIG_DIR="experiment_configs/video"
	;;
  real)
	CONFIG="transformer_real"
	CONFIG_DIR="experiment_configs/real"
	;;
  swapcups_normal)
	CONFIG="transformer_swapcups_normal"
	CONFIG_DIR="experiment_configs/real"
	;;
  swapcups_abs)
	CONFIG="transformer_swapcups_abs"
	CONFIG_DIR="experiment_configs/real"
	;;
  twoscoops_normal)
	CONFIG="transformer_twoscoops_normal"
	CONFIG_DIR="experiment_configs/real"
	;;
  real100)
	CONFIG="transformer_pickandplace_real_100"
	CONFIG_DIR="experiment_configs/real"
	;;
  pickandplace_real)
	CONFIG="transformer_pickandplace_real"
	CONFIG_DIR="experiment_configs/real"
	;;
  pickandplace_abs)
	CONFIG="transformer_pickandplace_abs"
	CONFIG_DIR="experiment_configs/real"
	;;
  pickandplace_normal)
	CONFIG="transformer_pickandplace_normal"
	CONFIG_DIR="experiment_configs/real"
	;;
  real299)
	CONFIG="transformer_real299"
	CONFIG_DIR="experiment_configs/real"
	;;
  real799)
	CONFIG="transformer_real799"
	CONFIG_DIR="experiment_configs/real"
	;;
  pusht)
	CONFIG="transformer_pusht"
	CONFIG_DIR="experiment_configs"
	  ;;
  *)
	echo "Invalid config_choice: $config_choice. Use a valid configuration."
	exit 1
	;;
esac

if [ "$cached" = "true" ]; then
    extra_command="policy.use_embed_if_present=true"
fi
if [ "$vjepa2" = "true" ]; then
    CONFIG="${CONFIG}_vjepa2"
fi
if [ "$emb" = "true" ]; then
    CONFIG="${CONFIG}_emb"
fi
hhmmss=$(date +%H%M%S)
hydra_command="hydra.run.dir=\"data/outputs/transptp-${past_action_pred}_s${past_steps_reg}_o${global_obs}_a${global_action}_${config_choice}_${hhmmss}_${emb}\""

echo $CONFIG
# Get the current time in HHMMSS format

# Calculate logging name
obs_len=$((global_obs))
action_pred=$((global_action))
pred_type="epsilon" # change if you want to test different targets

# Change training.debug if you want to test
# Construct and execute the command
echo "============================================================"
echo "Parameters:"
echo "--config-dir=${CONFIG_DIR}"
echo "--config-name=${CONFIG}"
echo "training.seed=42"
echo "policy.noise_scheduler.prediction_type=${pred_type}"
echo "training.device=cuda:0"
echo "${hydra_command}"
echo "global_obs=${global_obs}"
echo "global_action=${global_action}"
echo "global_horizon=${global_horizon}"
echo "policy.past_action_pred=${past_action_pred}"
echo "policy.past_steps_reg=${past_steps_reg}"
echo "training.debug=false"
echo "logging.name=transptp-${past_action_pred}_s${past_steps_reg}_o${global_obs}_a${global_action}_${config_choice}_${hhmmss}_${emb}"
echo "${extra_command}"
echo "============================================================"
WANDB_DISABLED=true python train.py \
	--config-dir="${CONFIG_DIR}" \
	--config-name="${CONFIG}" \
	training.seed=42 \
	policy.noise_scheduler.prediction_type=${pred_type}\
	training.device=cuda:0 \
    ${hydra_command}\
	global_obs=${global_obs} \
	global_action=${global_action} \
	global_horizon=${global_horizon} \
	policy.past_action_pred=${past_action_pred} \
	policy.past_steps_reg=${past_steps_reg} \
	training.debug=false \
	logging.name="transptp-${past_action_pred}_s${past_steps_reg}_o${global_obs}_a${global_action}_${config_choice}_${hhmmss}_${emb}" \
	${extra_command}

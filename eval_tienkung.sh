#!/bin/bash
# Usage: bash eval_tienkung.sh <experiment_id> <device>
# Example: bash eval_tienkung.sh 0330_tienkung_twist2 cuda:0

# You can specify a specific motion file here to evaluate on
motion_file="/data/gmr_dataset/ACCAD_GMR/Female1Walking_c3d/B10_-_walk_turn_left_(45)_stageii.pkl"

task_name="tienkung_stu_future"
proj_name="tienkung_stu_future"
exptid=$1
device=$2

cd legged_gym/legged_gym/scripts

echo "Evaluating Tienkung student policy with future motion support..."
echo "Task: ${task_name}"
echo "Project: ${proj_name}"
echo "Experiment ID: ${exptid}"
echo ""

# Run the evaluation script
python play.py --task "${task_name}" \
               --proj_name "${proj_name}" \
               --teacher_exptid "None" \
               --exptid "${exptid}" \
               --num_envs 1 \
               --device "${device}" \
               --motion.motion_file "${motion_file}" \
               # --checkpoint 1000 \
               # --record_video \

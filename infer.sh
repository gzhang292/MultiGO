#!/bin/bash

# 设置默认参数
INPUT_DIR="./ood_1"
TEMP_DIR1="${INPUT_DIR}_rmbg"
TEMP_DIR2="${INPUT_DIR}_rmbg_cropped_persons"
OUTPUT_DIR="${INPUT_DIR}_rmbg_cropped_persons_centralized"
CANVAS_SIZE=1024
HEIGHT_RATIO=0.85

# mkdir -p "$TEMP_DIR1" "$TEMP_DIR2" "$OUTPUT_DIR"

# echo "Step 1: Removing backgrounds..."
# python rmbg_a.py -i "$INPUT_DIR" -o "$TEMP_DIR1"

# echo "Step 2: Cropping persons..."
# python crop_person.py -i "$TEMP_DIR1" -o "$TEMP_DIR2"

# echo "Step 3: Centralizing images..."
# python centralize_rgba.py -i "$TEMP_DIR2" -o "$OUTPUT_DIR" --size "$CANVAS_SIZE" --ratio "$HEIGHT_RATIO"

# echo "Processing complete!"
# echo "Final results are saved in: $OUTPUT_DIR" 


# echo "Step 4: OpenPose..."

# cd ../openpose
# CUDA_VISIBLE_DEVICES=6 ./build/examples/openpose/openpose.bin --image_dir "../project/${OUTPUT_DIR}" --write_json "../project/${OUTPUT_DIR}" --display 0 --net_resolution -1x544 --scale_number 3 --scale_gap 0.25 --hand --face --render_pose 0
# cd ../project

# echo "Step 5: SMPL estimation..."

IMG_DIR=".${OUTPUT_DIR}"
SMPL_EST_DIR=".${OUTPUT_DIR}_estsmplx"

# cd smpl_estimated_related

# CUDA_VISIBLE_DEVICES=6 python fit.py \
#     --opt_orient \
#     --opt_betas \
#     -i "${IMG_DIR}" \
#     -o "${SMPL_EST_DIR}"



echo "Step 6: multigo inference..."

# Model paths
MODEL_PATH="./workspace/model.safetensors"
WORKSPACE_DIR="../${INPUT_DIR}_recontructed_gaussians"


cd ./multigo


CUDA_VISIBLE_DEVICES=3 python infer.py big \
    --resume "${MODEL_PATH}" \
    --workspace "${WORKSPACE_DIR}" \
    --infer_img_path "${IMG_DIR}" \
    --infer_smpl_path "${SMPL_EST_DIR}"



echo "Step 7: Convert glb..."

bash converts.sh ../${INPUT_DIR}_recontructed_gaussians/ood


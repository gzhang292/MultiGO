

directory=$1

export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Loop through the files in the directory
for file in $directory/*.ply; do

    {
        number=$(basename "$file" .ply)
        dir=${directory}/${number}.glb
        if [ -f "$dir" ]; then
            continue
        fi
        # 构建命令，替换参数中的 "${number}0" 为当前循环中的数字
        command="python convert.py big --test_path ${directory}/${number}.ply"
        # 打印并执行命令
        echo "Running command: $command"
        CUDA_VISIBLE_DEVICES=7 $command

    } 
    sleep 1

done

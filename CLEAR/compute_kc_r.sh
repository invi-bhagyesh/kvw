CUDA_VISIBLE_DEVICES=0 python3 -m baselines.KVW \
	--model_id Qwen/Qwen2-VL-2B-Instruct \
	--vanilla_dir checkpoints/qwen2B_vanilla \
	--forget_ratio 05 \
	--batch_size 4 \
	--num_epochs 1 \
	--phase compute_kc_r \
	--data_folder data/CLEAR \
	--save_dir /workspace/mmu/checkpoints/temp/1
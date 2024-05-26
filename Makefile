docker-build:
	sudo docker build \
		--build-arg WANDB_TOKEN=${WANDB_TOKEN} \
		--build-arg HF_TOKEN=${HF_TOKEN} \
		-t pldiffusion:latest .

docker-build-rmi:
	sudo docker rmi localhost/pldiffusion:latest \
		&& sudo docker build \
			--build-arg WANDB_TOKEN=${WANDB_TOKEN} \
			--build-arg HF_TOKEN=${HF_TOKEN} \
			-t pldiffusion:latest .

docker-run:
	sudo docker run --rm -it \
		--shm-size=256g \
		--device nvidia.com/gpu=all  \
		-v /mnt/beegfs/mario.dejesusdagraca/bfscc:/data/.cache/bfscc \
		-v /mnt/beegfs/mario.dejesusdagraca/bfscc_filtered:/data/.cache/bfscc_filtered \
		-v /mnt/beegfs/mario.dejesusdagraca/wandb:/data/.cache/wandb \
		-v /mnt/beegfs/mario.dejesusdagraca/checkpoints:/data/.cache/checkpoints \
		-v /mnt/beegfs/mario.dejesusdagraca/final_samples:/data/.cache/final_samples \
		-v /home/mario.dejesusdagraca/thesis-research/src:/data/src \
		pldiffusion:latest
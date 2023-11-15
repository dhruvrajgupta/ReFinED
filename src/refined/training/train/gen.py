# Open a file in write mode ('w')
lang = "ru"
typex = "train"

for i in range(10, 101, 10):
    content=f"""#!/bin/bash
#SBATCH -p gpu_8 # partition (queue)
#SBATCH --mem 70000 # memory pool for all cores (180GB)
# SBATCH --mem 16000 # memory pool for all cores (16GB)
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH -c 2 # number of cores
#SBATCH -o log/%x.%N.%j.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e log/%x.%N.%j.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J {lang}_{i}p_{typex} # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=BEGIN,END,FAIL # (recive mails about end and timeouts/crashes of your job)
#SBATCH --mail-user=dhruv.learner@gmail.com
# Print some information about the job to STDOUT
source ~/.bashrc
conda activate refined38
# source activate virtualEnvironment
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
# Job to perform
# curl https://ia801807.us.archive.org/25/items/wikibase-wikidatawiki-20210201/wikibase-wikidatawiki-20210201_files.xml --output abcd.xml
export WANDB_API_KEY=3e09083601b542ac185074d2f0b366568563e4de
python fine_tune.py --experiment_name {lang}_{i}p_{typex} --ds_percent {i}p --language {lang} --resume False

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
"""
    with open(f'{lang}_{i}p_{typex}.sb', 'w') as file:
        # Write content to the file
        file.write(content)
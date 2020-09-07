for FEAT_NAME in `cat data/zero_shot/train_feats.config_file.8_random_feats.tsv | sed -e 1d | awk '{print $1}' | sed -e s/\"//g`
do
	sbatch -J $FEAT_NAME -p gpu --mem 32gb --gres gpu:1 -c 3 --time 2-0 --wrap "/usr/bin/time -v python -u amber_single_run.py --wd outputs/$FEAT_NAME --feat-name $FEAT_NAME"
done

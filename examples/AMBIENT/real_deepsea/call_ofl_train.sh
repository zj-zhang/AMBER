# call offline training script
#for ds in 0.3 0.5 0.7 0.9; do
for ds in 0.1 0.2 0.05 0.01; do
    for i in `seq 0 4`; do
	echo $ds $i
	mkdir -p ./outputs/ds-$ds/rep_$i
        python train_from_history.py --analysis train --downsample $ds --wd ./outputs/ds-$ds/rep_$i > ./outputs/ds-$ds/rep_$i/log.txt
        python train_from_history.py --analysis reload --downsample $ds --wd ./outputs/ds-$ds/rep_$i
    done
done

#for i in `seq 0 4`; do
#	echo $i
#	mkdir -p ./outputs/full/rep_$i
#	python train_from_history.py --analysis train --wd ./outputs/full/rep_$i > ./outputs/full/rep_$i/log.txt
#	python train_from_history.py --analysis reload --wd ./outputs/full/rep_$i
#done


#python train.py --data_root /backup/home/ylb/data --name debug --batch_size 4 --max_dataset_size 32000 --niter 20 --niter_decay 40 --save_result_freq 250 --save_epoch_freq 5 --ndown 6
python train.py --data_root /backup/home/ylb/data --name debug --model vggnet --netD conv-up --batch_size 4 --max_dataset_size 32000 --niter 20 --niter_decay 40 --save_result_freq 250 --save_epoch_freq 5 --ndown 6

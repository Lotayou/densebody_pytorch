# Prepare Datasets and UV map Labels

### Prerequisites
```
Windows 10 / Ubuntu 18.04
CUDA 9.0 / 9.1
Python 3.6
PyTorch 1.0.0
opencv-python
tqdm
chumpy (For converting SMPL model to basic numpy arrays)
h5py (For processing Human36m annotations)
```

Make sure you have an GPU card with at least 12 GB graphic memory. You can decrease batch size too, but the performance is not guaranteed.

(Optional) Linux users are encouraged to install [torch-batched-svd](https://github.com/KinglittleQ/torch-batch-svd) for speedup.

### Prepare dataset
We hereby datail the necessary steps to reproduce human36m experiments. Most people failed on this step, so I hope this instruction could make sure you don't become one of them.

1. I tentatively suggest users to first try download human36m dataset from the [official website](http://vision.imar.ro/human3.6m/description.php). However manual authorization could take anywhere from 6 days to 6 months, so if you are keen to get things going, you can start by playing with this [toy_dataset](https://pan.baidu.com/s/1szhb9B_8n6p6CeAoPUxnhw). Extraction code: 0o95 
Alternative link [Google Drive](https://drive.google.com/open?id=1ssuUje20x1PS5qYwbg1AAloDsVwF1eTW)

2. Unpack downloaded zip files into a single folder (which I suggest you name it `human36m`) and put it under `path-to-your-datasets`. 
Open `data_utils/data_washing.py`, change the `root_dir` variable in main function near line 190 to your h36m dataset path and run it to perform standard data augmentations, the washed dataset will be stored in `path-to-your-datasets/human36m_washed`.

### Create UV maps
3. To acquire SMPL's default UV map, follow these steps:
- Register on SMPL's official [website](http://smpl.is.tue.mpg.de) and download their maya fbx animation files, store them under `data_utils`.
- Install [Blender](https://www.blender.org/) and open the IDE, import a `fbx` file inside Blender and then export it as `.obj` format.
- Run `data_utils/triangulation.py` to convert Blender converted obj file into SMPL compatible one (with 13776 triangulated faces).

If you are interested, you can also try with [radvani's UV map](https://github.com/Lotayou/densebody_pytorch/issues/4#issuecomment-481480724) attached in the [issues](https://github.com/Lotayou/densebody_pytorch/issues/4). Incidentally, you may want to read this issue carefully for more information in case you get stucked midway.

4. After you acquired the converted .obj file, put it under `data_utils` and rename it as `'{}_template.obj'.format(opt.uv_map)` where options can be found in `train.py` (default is `radvani`). Run `data_utils/uv_map_generator.py` to calculate barycentric interpolation weights and other necessary caching informations. This could take 8~10 minutes.

5. Processing SMPL model with `data_utils/preprocess_smpl.py`. You can also check the instructions [here](https://github.com/Lotayou/SMPL).

6. Run `data_utils/create_UV_maps` to create UV labels. This takes 5 hours on my PC, you can monitor the progress through tqdm info. The processed UV maps will be stored at `path-to-your-datasets/human36m_UV_map_{}` where `{}` again is `opt.uv_map`.

### Training
7. Go back to the root folder, checkout training options in `train.py` and modify them in `train.sh`, specify your gpu id through `os.environ['CUDA_VISIBLE_DEVICES']`, run `bash train.sh`, and Bob's your uncle:)

The training script is designed to monitor losses in realtime through `tqdm` progress bar. However, if you are working on a remote server or cluster via ssh connections, it's better to submit your job with `bash nohup_train.sh`. This script replaced tqdm monitoring with a `log.txt` file under your checkpoint folder where loss info is printed every `opt.save_result_freq` batches.

### Testing
We are still working on the training part. The testing scripts will be released later.

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

(Optional) Linux users are encouraged to install [torch-batched-svd](https://github.com/KinglittleQ/torch-batch-svd) for speedup.

### Prepare dataset
We hereby datail the necessary steps to reproduce human36m experiments. Most people failed on this step, but this instruction will make sure you ain't one of them.

1. Download human36m dataset from the [official website](http://vision.imar.ro/human3.6m/description.php)... Nah I'm just kidding. The admins of that website are too damn slow, you'll be lucky if they'd approve your account in six months. Instead, I suggest you check out [this repo](https://github.com/MandyMo/pytorch_HMR) (wink-wink...)

2. Unpack downloaded zip files into a single folder (which I suggest you name it `human36m`) and put it under `path-to-your-datasets`. 

Open `data_utils/data_washing.py`, change the `root_dir` variable in main function near line 190 to your h36m dataset path and run it to perform standard data augmentations, the washed dataset will be stored in `path-to-your-datasets/human36m_washed`.

### Create UV maps
3. Get yourself a nice UV map for SMPL meshes. You can either refer to [radvani's UV map](https://github.com/Lotayou/densebody_pytorch/issues/4#issuecomment-481480724) or convert from SMPL's official .fbx files using [FBX-SDK](https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2019-0). Keep in mind that the conversion could be very tricky though. You are encoraged to read the [issues](https://github.com/Lotayou/densebody_pytorch/issues/4) for more information anytime you get stucked.

4. After you acquired the converted .obj file, put it under `data_utils` and rename it as `'{}_template.obj'.format(opt.uv_map)` where options can be found in `train.py` (default is `radvani`). Run `data_utils/uv_map_generator.py` to calculate barycentric interpolation weights and other necessary caching informations. This could take 8~10 minutes.

5. Run `data_utils/create_UV_maps` to create UV labels. This takes 5 hours on my PC, you can monitor the progress through tqdm info. The processed UV maps will be stored at `path-to-your-datasets/human36m_UV_map_{}` where `{}` again is `opt.uv_map`.

### Training
6. Go back to the root folder, checkout training options in `train.py` and modify them in `train.sh`, specify your gpu id through `os.environ['CUDA_VISIBLE_DEVICES']`, run `bash train.sh`, and Bob's your uncle:)

### Testing
We are still working on the training part. The testing scripts will be released later.

# densebody_pytorch
PyTorch implementation of CloudWalk's recent paper [DenseBody](https://arxiv.org/abs/1903.10153v3).

**Note**: For most recent updates, please check out the `dev` branch.

**Update on 20190613** A toy dataset has been released to facilitate the reproduction of this project. checkout [`PREPS.md`](PREPS.md) for details.

**Update on 20190826** A pre-trained model ([Encoder](https://yadi.sk/d/isvVFGIU6cHueQ)/[Decoder](https://yadi.sk/d/ck-JBue4XYNpHQ)) has been released to facilitate the reproduction of this project. 

![paper teaser](teaser/teaser.jpg)

### Reproduction results

Here is the reproduction result (left: input image; middle: ground truth UV position map; right: estimated UV position map)

<div align="center">
  <img src="https://user-images.githubusercontent.com/33449901/56275710-cce07800-6133-11e9-9507-cfc347a51006.png" width="800px" />
</div>

### Update Notes
- SMPL official UV map is now supported! Please checkout [`PREPS.md`](PREPS.md) for details.
- Code reformating complete! Please refer to `data_utils/UV_map_generator.py` for more details.
- Thanks [Raj Advani](https://github.com/radvani) for providing new hand crafted UV maps!

### Training Guidelines
Please follow the instructions [`PREPS.md`](PREPS.md) to prepare your training dataset and UV maps. Then run `train.sh` or `nohup_train.sh` to begin training. 

### Customizations

To train with your own UV map, checkout [`UV_MAPS.md`](UV_MAPS.md) for detailed instructions.

To explore different network architectures, checkout [`NETWORKS.md`](NETWORKS.md) for detailed instructions.

### TODO List
- [x] Creating ground truth UV position maps for Human36m dataset.
    - [x] [20190329]() Finish UV data processing.
    - [x] [20190331]() Align SMPL mesh with input image.
    - [x] [20190404]() Data washing: Image resize to 256*256 and 2D annotation compensation.
    - [x] [20190411]() Generate and save UV position map.
        - [x] [radvani](https://github.com/radvani) Hand parsed new 3D UV data
        - [x] Validity checked with minor artifacts (see results below)
        - [x] Making UV_map generation module a separate class.
    - [x] [20190413]() Prepare ground truth UV maps for washed dataset.
    - [x] [20190417]() SMPL official UV map supported!
    - [x] [20190613]() A testing toy dataset has been released!
    
- [x] Prepare baseline model training
    - [x] [20190414]() Network design, configs, trainer and dataloader
    - [x] [20190414]() Baseline complete with first-hand results. Something issue still needs to be addressed.
    - [x] [20190420]() Testing with different UV maps.

### Authors
**[Lingbo Yang(Lotayou)](https://github.com/Lotayou)**: The owner and maintainer of this repo.

**[Raj Advani(radvani)](https://github.com/radvani)**: Provide several hand-crafted UV maps and many constructive feedbacks.

### Citation
Please consider citing the following paper if you find this project useful.

[DenseBody: Directly Regressing Dense 3D Human Pose and Shape From a Single Color Image](https://arxiv.org/abs/1903.10153v3)

### Acknowledgements
The network training part is inspired by [BicycleGAN](https://github.com/junyanz/BicycleGAN)

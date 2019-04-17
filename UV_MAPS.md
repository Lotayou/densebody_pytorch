# Use your customized UV map

### Motivations
Currently we have explored various UV maps, and found out some particular artifacts that we believe is closely related to your choice of UV texture mappings. For example, we observed frequent color inconsistency when testing with `radvani` UV maps as showcased here (note the colors of both hands on bottom-right corner):

<div align="center">
  <img src="https://user-images.githubusercontent.com/33449901/56272849-2a71c600-612e-11e9-928f-e03c365f4e9c.png" width="800px" />
</div>

We believe the segmentation of body parts and the placement of UV atlases plays an important role in training, but so far we haven't reached a solid conclusion yet. Therefore we encourage the users to explore and report results trained with their own UV maps. This document hereby details the necessary steps to use your customized UV maps.

### Instructions

To create a UV map, you need to load an SMPL template model into any 3D animation software like MAYA or Blender. Then follow necessary steps to cut the mesh into different parts and unfold each part onto an UV grid map. Then assign 3D vertex to UV vertex correspondences and UV texture coordinates between [0,1]. 

To use your UV map for our projects, save your UV data as a single .obj file, name it `{your_uv}_template.obj`. Put it under `data_utils` folder and follow the instructions in [`PREPS.md`](PREPS.md) to prepare datasets and labels.
But before that, it's highly recommended to check if your UV info is correct. You can run `data_utils\uv_map_generator_unit_test.py` for a quick check, it reads in first 10 items of your dataset (I'm using human36m) and create corresponding UV labels inside `_test_cache` folder. 

The correct UV maps should look like this:

<div align="center">
  <img src="https://user-images.githubusercontent.com/33449901/56273845-2d6db600-6130-11e9-817b-0774f71cc4ec.png" width="800px" />
</div>

While an incorrect UV map could look like this (notice the color inconsistency inside each atlas):

<div align="center">
  <img src="https://user-images.githubusercontent.com/33449901/56273865-36f71e00-6130-11e9-913d-32a036ebff1d.png" width="800px" />
</div>

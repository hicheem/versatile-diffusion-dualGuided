# README

## Important Files & Implementation:

### Hugging Face Imports:
Several important files from Diffusers' Hugging Face library have been integrated into the project:
- `attention.py`
- `diffusion_utils.py`
- `openaimodel.py` which has been renamed to `models.py`
  
### Model Implementation:
Inside `models.py`, the implementation for the Dual Guided model can be found under the class name `UNetModelVD_DualGuided`.

### Training Notebook:
There's an accompanying `dual_guided_versatile.ipynb` notebook file that provides a comprehensive workflow to train the model.

## Note on Memory Management:

During the development of this project, I leveraged `float16` precision and cast models to `.half()` to counteract RAM limitations. Despite these precautions, training the model directly faced challenges due to its extensive parameter set. Specifically, while the model's forward pass and loss calculations worked without hitches, the `loss.backward()` operation couldn't be executed due to RAM overheads.

However, the sampling function was successfully implemented and tested using pretrained weights.

# Epiretinal Membrane Detection in Optical Coherence Tomography Retinal Images using Deep Learning
Article link: https://ieeexplore.ieee.org/document/9477630

This is a repository for the trained models to detect ERM in OCT B-scans, the best model for each architecture.
 
### Architectures:
<ul>
  <li>AlexNet</li>
  <li>SqueezeNet</li>
  <li>ResNet-101</li>
  <li>VGG-19</li>
</ul>

All models were trained and tested using:
<ul>
  <li>PyTorch 1.5.0+cu101</li>
  <li>torchvision 0.6.0+cu101</li>
</ul>

### Usage

In each sub-folder we provide the model definition (net.py), the pre-trained weights (weights.pth), and a basic example on how to use these models to assign a label to a single image (test-code.py).

Sample images source: http://theretinagroup.com/epiretinal-membrane/

### Citation

```
@article{9477630,
  author={Parra-Mora, Esther and Cazañas-Gordon, Alex and Proença, Rui and Da Silva Cruz, Luís A.},
  journal={IEEE Access}, 
  title={Epiretinal Membrane Detection in Optical Coherence Tomography Retinal Images using Deep Learning}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/ACCESS.2021.3095655}
 }
```

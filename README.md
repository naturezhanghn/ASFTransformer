### Alternating Spatial-Frequency Transformer
ASF-Transformer: neutralizing the impact of atmospheric turbulence on optical imaging through alternating learning in the spatial and frequency domains
[paper link](https://opg.optica.org/oe/viewmedia.cfm?uri=oe-31-22-37128&seq=0)
<br>
Atmospheric turbulence is a complex phenomenon that poses challenges in optical imaging, particularly in applications like astronomy, remote sensing, and surveillance. The ASF-Transformer is designed to tackle this challenge head-on.

#### Key Features:
- **Alternating Learning in Spatial and Frequency Domains (LASF) Mechanism**: Inspired by the principles of split-step propagation and correlated imaging, ASF-Transformer includes the LASF mechanism, which alternately implements self-attention in both spatial and Fourier domains.
- **Enhanced Texture Recovery**: Assisted by Patch FFT loss, the ASF-Transformer can recover intricate textures without the need for generative adversarial schemes.
- **State-of-the-art Performance**: Evaluations across diverse test mediums show the model's superior performance compared to recent turbulence removal methods.

#### Benefits:
- **Novel Approach**: Unlike conventional GAN-based solutions, the ASF-Transformer opens a new pathway for handling real-world image degradations.
- **Insights into Neural Network Design**: By incorporating principles from optical theory, the ASF-Transformer not only provides a solution for turbulence mitigation but also offers potential insights for future neural network design.

#### How to prepare the dataset：
```
dataset/
│   └── nature_turbdata/
│       ├── algorithm_simulated_videos/
│       │   ├── test/
│       │   │   ├── *.png
│       │   │   └── *turb.png
│       │   ├── train/
│       │   │   ├── *.png
│       │   │   └── *turb.png
│       │   └── val/
│       │       ├── *.png
│       │       └── *turb.png
│       └── physical_simulated_videos/
│           ├── test/
│           │   ├── *.png
│           │   └── *turb.png
│           ├── train/
│           │   ├── *.png
│           │   └── *turb.png
│           └── val/
│               ├── *.png
│               └── *turb.png
```




#### How to Use:
1. Install the required Python libraries: `pip install -r requirements.txt`.
2. Modify the configuration files ending in `.yml` located in `./Turbulence/Options/`.
3. Update `run.sh` to replace the path with the new `.yml` configuration file.
4. Execute the file by running `sh run.sh`.

#### Presentation Slides:
<center>
<img src="https://github.com/naturezhanghn/ASFTransformer/assets/71700470/fde7a6a0-58e7-4fe1-bb5d-7b7cb1199818" width="600">  
<br>
<img src="https://github.com/naturezhanghn/ASFTransformer/assets/71700470/f33fbc94-ab68-4b6d-b1d1-da3aafb0452d" width="650">   
<br>
<img src="https://github.com/naturezhanghn/ASFTransformer/assets/71700470/d7d1bcef-bfe3-49ef-9dd9-16dcaa77c6d6" width="500">   
<br>
<img src="https://github.com/naturezhanghn/ASFTransformer/assets/71700470/0f1891fa-0739-467f-8fc7-2846cc60bd2a" width="450">  
</center>




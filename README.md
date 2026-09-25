<h2 align="center">SymPoint Revolutionized: Boosting Panoptic Symbol Spotting with Layer Feature Enhancement</h2>
<p align="center">
  <img src="assets/framework.png" width="75%">
</p>



## 🔧Installation & Dataset
#### Environment

We recommend users to use `conda` to install the running environment. The following dependencies are required:

```bash
conda init
conda create -n spv1 python=3.9 -y
conda activate spv1
conda install -y -c conda-forge ninja
conda install -y -c nvidia/label/cuda-13.0.0 cuda-toolkit

pip install torch torchvision torchaudio
pip install gdown mmcv==0.2.14 svgpathtools==1.6.1 munch==2.5.0 tensorboard==2.12.0 tensorboardx==2.5.1 

git clone https://github.com/facebookresearch/detectron2.git && cd detectron2 
python -m pip install .

# compile pointops
cd modules/pointops
python setup.py install
```

#### Dataset&Preprocess

download dataset from floorplan website, and convert it to json format data for training and testing.

```python
# download dataset
python dataset/download_data.py
# preprocess
#train, val, test
python parse_svg.py --split train --data_dir ./dataset/train/train/svg_gt/
python parse_svg.py --split val --data_dir ./dataset/val/val/svg_gt/
python parse_svg.py --split test --data_dir ./dataset/test/test/svg_gt/
```

## 🚀Quick Start

```
#train
bash tools/train_dist.sh
#test
bash tools/test_dist.sh
```
## model weights
I uploaded the weight in google drive, you can download from this url: https://drive.google.com/file/d/1ZeWtgZJKD_yWmFNWwBOMN9_4-x-ZXUuS/view?usp=drive_link


## 📌Citation
If you find our paper and code useful in your research, please consider giving a star and citation.
<pre><code>
@article{liu2024sympoint,
  title={SymPoint Revolutionized: Boosting Panoptic Symbol Spotting with Layer Feature Enhancement},
  author={Liu, Wenlong and Yang, Tianyu and Yu, Qizhi and Zhang, Lei},
  journal={arXiv preprint arXiv:2407.01928},
  year={2024}
}
</code></pre>

pip uninstall -y opencv-python
pip install opencv-python-headless
(spv2) ubuntu@thunder-client:~/SymPointV2$ PYTHONPATH=./ python tools/inference.py models/spv2-rep/svg_pointT.yaml models/spv2-rep/best.pth --datadir dataset/custom/ --split val --type PDF --cvat_xml dataset/custom/annotations.xml --cvat_img_dir dataset/custom/
(spv2) ubuntu@thunder-client:~/SymPointV2/modules/pointops$ rm -rf build *.egg-info
(spv2) ubuntu@thunder-client:~/SymPointV2/modules/pointops$ pip uninstall -y pointops
WARNING: Skipping pointops as it is not installed.
(spv2) ubuntu@thunder-client:~/SymPointV2/modules/pointops$ echo $TORCH_CUDA_ARCH_LIST
(spv2) ubuntu@thunder-client:~/SymPointV2/modules/pointops$ sed -i 's/best_dist\[100\]/best_dist[256]/; s/best_idx\[100\]/best_idx[256]/' src/knnquery/knnquery_cuda_kernel.cu
(spv2) ubuntu@thunder-client:~/SymPointV2/modules/pointops$ grep -n "best_dist\[2\|best_idx\[2" src/knnquery/knnquery_cuda_kernel.cu


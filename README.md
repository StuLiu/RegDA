<h2 align="center">Learn from Segment Anything Model: Local Region Homogenizing for Cross-domain 
Remote Sensing Image Segmentation</h2>

<h5 align="center">
  by <a href="https://scholar.google.com/citations?user=LXlWdyQAAAAJ&hl=zh-CN">Wang Liu</a>,
  <a href="https://scholar.google.com/citations?hl=zh-CN&user=IYUlx_8AAAAJ">Puhong Duan</a>, 
  <a href="https://www.researchgate.net/profile/Zhuojun-Xie-6">Zhuojun Xie</a>, 
  <a href="https://scholar.google.com/citations?hl=zh-CN&user=5XOeLZYAAAAJ">Xudong Kang</a>, 
  <a href="https://scholar.google.com/citations?user=PlBq8n8AAAAJ&hl=zh-CN">Shutao Li</a>
</h5>


<h5 align="left">This is the official implementation for 
<a href="https://ieeexplore.ieee.org/abstract/document/10642007">RegDA</a>.
  
---------------------

<div align=center><img src="asserts/overview-regda.png"></div>
<p align="center">Fig. 1 An overview of the proposed RegDA.</p>


## Getting Started

### Environment:
- conda create -n regda python=3.8
- source activate regda
- pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
- conda install pytorch-scatter -c pyg
- pip install -r requirement.txt
- pip install -e .

### Prepare datasets

#### 1. Generate from raw:

- Download the raw datasets <a href="https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx">here</a>.
- Run the preprocess script in ./convert_datasets/ to crop train, val, test sets:\
`python convert_datasets/convert_potsdam.py`\
`python convert_datasets/convert_vaihingen.py`
- Generate local regions by running \
`python tools/seg_everything.py`
#### 2. Alternatively, you can also download the processed data
- Download the processed datasets <a href="https://pan.baidu.com/s/1rWHSgRpSVPlLt5_bykHCOg?pwd=6th5">here</a>.
- reorganize the directory tree.
#### 3. The prepared data is formatted as follows:
"\
-&nbsp;data\
&nbsp;&nbsp;&nbsp;|---&nbsp;IsprsDA\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---Potsdam\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---&nbsp;ann_dir\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---&nbsp;img_dir\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---&nbsp;reg_dir\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---Vaihingen\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---&nbsp;ann_dir\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---&nbsp;img_dir\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---&nbsp;reg_dir\
"

### Evaluate the trained RegDA models.\
Download the pre-trained [<b>weights</b>](https://pan.baidu.com/s/1rWHSgRpSVPlLt5_bykHCOg?pwd=6th5) and logs.
#### 1. on Vaihingen (IRRG) -> Potsdam (IRRG) task
Run evaluating: 
```python
python tools/eval.py --config-path st.regda.2potsdam --ckpt-path log/regda/2potsdam/ssl/Potsdam_best.pth --test 1
```
#### 2. on Potsdam (IRRG) -> Vaihingen (IRRG) task
Run evaluating: 
```python
python tools/eval.py --config-path st.regda.2vaihingen --ckpt-path log/regda/2vaihingen/ssl/Vaihingen_best.pth --test 1
```

### Train the RegDA
```bash 
bash runs/regda/run_2potsdam.sh
```
```bash 
bash runs/regda/run_2vaihingen.sh
```

### Inference single file
```bash 
python tools/infer_single.py st.regda.2potsdam log/regda/ssl/Potsdam_best.pth [image-path] --save-dir [save-dir-path]
```

### Cite
If you are interested in this work, please cite it as follows:
```bibtxt
@INPROCEEDINGS{10642007,
  author={Liu, Wang and Duan, Puhong and Xie, Zhuojun and Kang, Xudong and Li, Shutao},
  booktitle={IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium}, 
  title={Learn From Segment Anything Model: Local Region Homogenizing for Cross-Domain Remote Sensing Image Segmentation}, 
  year={2024},
  pages={8351-8354},
  doi={10.1109/IGARSS53475.2024.10642007}
}
```

---------------------

There are some differences compared with the published paper.</h5>

- The denoising approach in this repository is re-weighting, while in the paper is voting;
- A class-frequency threshold is utilized to guide the pseudo-label homogenizing. 
For each local region, if the class with a larger frequency than this threshold, 
it will be employed for homogenizing;
- The LRH is also utilized in the aligning stage.

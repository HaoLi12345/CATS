# CATS:COMPLEMENTARY CNN AND TRANSFORMER ENCODERS FOR SEGMENTATION

```
@inproceedings{li2022cats,
  title={Cats: Complementary CNN and Transformer Encoders for Segmentation},
  author={Li, Hao and Hu, Dewei and Liu, Han and Wang, Jiacheng and Oguz, Ipek},
  booktitle={2022 IEEE 19th International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2022},
  organization={IEEE}
}
```


# CATS v2: Hybrid encoders for robust medical segmentation

```
@article{li2023cats,
  title={CATS v2: Hybrid encoders for robust medical segmentation},
  author={Li, Hao and Liu, Han and Hu, Dewei and Yao, Xing and Wang, Jiacheng and Oguz, Ipek},
  journal={arXiv preprint arXiv:2308.06377},
  year={2023}
}
```




train.py ------> train BTCV dataset with .jason file. (based on https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/unetr_btcv_segmentation_3d.ipynb)<br />


train_with_data_dir.py -----> changed data I/O, you can use a path that contains nifti files, in the following format<br />

dataset_folder<br />
└─train_set_image_folder <br />
&emsp;├── 1.nii.gz <br />
&emsp;└── 2.nii.gz <br />
&emsp;... <br />
&emsp;... <br />
&emsp;... <br />
&emsp;└── 100.nii.gz <br />
└─train_set_label_folder <br />
&emsp;├── 1_label.nii.gz <br />
&emsp;└── 2_label.nii.gz <br />
&emsp;... <br />
&emsp;... <br />
&emsp;... <br />
&emsp;└── 100_label.nii.gz <br />
		

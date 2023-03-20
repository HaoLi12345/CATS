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


train.py ------> train BTCV dataset with .jason file. (based on https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/unetr_btcv_segmentation_3d.ipynb)<br />


train_with_data_dir.py -----> changed data I/O, you can use a path that contains nifti files, in the following format<br />

────────dataset_folder<br />
	  |<br />
	  |<br />
	  ├─train_set_image_folder <br />
	  |  	├── 1.nii.gz <br />
	  |  	└── 2.nii.gz <br />
	  | 	... <br />
	  |  	... <br />
	  |  	... <br />
	  |  	... <br />
	  |  	└── 100.nii.gz (bunch of nifti files in a folder) <br />
	  |<br />
	  |<br />
	  |<br />
	  └─train_set_label_folder <br />
	  	├── 1_label.nii.gz <br />
	  	└── 2_label.nii.gz <br />
	        ... <br />
	        ... <br />
	        ... <br />
	        ... <br />
	        └── 100_label.nii.gz (bunch of nifti files in a folder) <br />
		

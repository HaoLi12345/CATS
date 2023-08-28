Added train_v2_moda.py for crossmoda dataset described in cats_v2. 

train_v2_moda.py is almost same as train.py but different model selection. 

You could use original train files :)

08/17/23

------------------------------------------------------------

you may need to delete line 75 (ToTensord(keys=["image"]),) from test.py if you are runing with latest monai. (ref link: https://colab.research.google.com/github/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d.ipynb#scrollTo=TS5HZojSuKI7)



comments_date: 03/21/23	

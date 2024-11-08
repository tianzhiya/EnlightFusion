# EnlightFusion
Code of EnlightFusion:It is an infrared and visible light image fusion algorithm specifically designed for low-light environments.
## Tips:<br>
Due to file size issues, the training set has been removed from the code and the MSRS dataset can be downloaded here: https://pan.baidu.com/s/18q_3IEHKZ48YBy2PzsOtRQ?pwd=MSRS
Place the downloaded training dataset under: MSRS/ path.

## To Train
Run "**python trainEnlightFusion.py**" to train the model.
The training data are selected from the MSRS dataset. 

## To Test
Please download the pre-trained model from the link https://pan.baidu.com/s/1qfOx5VlPGg9iB8d7m5puFA?pwd=b593  using the extraction code b593. After downloading, perform the testing with the model. Subsequently, place the downloaded model file in the directory /model/Fusion/.

Run "**python TestEnlightFusion.py**" to test the model.
The images generated by the test will be placed under the lianHeXuLian/EnlightOut/MSRSFusion path.

If this work is helpful to you, please cite it as:
```
@article{EnlightFusion,
  title={EnlightFusion: A visual enhancement-based fusion
algorithm for infrared and visible images in
low-light environments},
author={Quanquan Xiao ,Haiyan jin,Haonan Su,etc},
}
```
If you have any question, please email to me (1211211001@stu.xaut.edu.cn).

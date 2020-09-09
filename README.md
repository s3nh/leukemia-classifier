#### Clown-fiesta ideas repo


#### 24-08 

README refactor 



```classification```

Main assumption of classification part is to prepare pipeline(?) which allow to train 
classification models regardless of its complexity, mainly based on config files. 
To afford this we used pytorch-lightning and assume that parameters should be stored in config.yaml 
file to avoid much complexity. 

To use classifier properly you should prepare your dataset in ImageFolder like structure. (TODO: Dataloder will be highly recommended)

```
Basic example:

data\
	train\
		class_1\
		class_0\
	valid\	
		class_1\
		class_0\

```

You can define path to your dataset in configuration files

At this stage there is ```resnet50``` pretrained model with 3 FC layers, 
which size you can describe in config file. There is also parameter with num classes which 
you want to predict.

#### Running 

to run training process properly you can use 

``` 

python3 main.py

```


which call for src/data.py, src.train.py and combine them in ```pytorch_lightning```-like pipeline 


tbc;



```rust-inference``` 

concept, maybe well want to use rust like backend so with little effort 
we can use some rust-tract with models converted to .onnx format


``` rust-rest-api```

rest api with file transfer and data storage functionalities (copied from some other repo)

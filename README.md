# PKU_Project







### Lifespan age transformation GANs 使用注意
由于[Lifespan Age Transformation Synthesis](https://github.com/royorel/Lifespan_Age_Transformation_Synthesis)项目并没有被集成到PyPi和Anaconda仓库中 ，不能直接使用```conda install```或者```pip install```安装，需要运行的python环境中下载GPU版本的pytorch，并且按照其仓库中的```README.txt```的说明下载依赖，并在本项目中的```Main.py```文件中的```generate_life_span_video()```的```script_directory```替换为```Lifespan Age Transformation Synthesis```项目在你的 local  machine 中的位置
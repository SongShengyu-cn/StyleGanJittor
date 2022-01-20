# StyleGanJittor (Tsinghua university computer graphics course)
# Overview
Jittor 64*64 implementation of StyleGAN (Tsinghua university computer graphics course)
This project is a repetition of StyleGAN based on python 3.8 + [Jittor（计图）](https://github.com/Jittor/jittor) and [The open source StyleGAN-Pytorch project](https://github.com/rosinality/style-based-gan-pytorch). I train the model on the color_symbol_7k dataset for 40000 iterations. The model can generate 64×64 symbolic images.


StyleGAN is a generative adversarial network for image generation proposed by NVIDIA in 2018. According to the [paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.html), the generator improves the state-of-the-art in terms of traditional distribution quality metrics, leads to demonstrably better interpolation properties, and also better disentangles the latent factors of variation. The main improvement of this network model over previous models is the structure of the generator, including the addition of an eight-layer Mapping Network, the use of the AdaIn module, and the introduction of image randomness - these structures allow the generator to The overall features of the image are decoupled from the local features to synthesize images with better effects; at the same time, the network also has better latent space interpolation effects.

(Karras T, Laine S, Aila T. A style-based generator architecture for generative adversarial networks[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 4401-4410.)

The training results are shown in [Video1trainingResult.avi](https://github.com/SongShengyu-cn/StyleGanJittor/blob/main/Video1trainingResult.avi), [Video2GenerationResult1.avi](https://github.com/SongShengyu-cn/StyleGanJittor/blob/main/Video2GenerationResult1.avi), and [Video3GenerationResul2t.avi](https://github.com/SongShengyu-cn/StyleGanJittor/blob/main/Video3GenerationResult2.avi) generated by the trained model.

The Checkpoint folder is the trained StyleGAN model, because it takes up a lot of storage space, only the '040000.model' obtained by the last training is kept, and the rest of the models have been deleted.The data folder is the color_symbol_7k dataset folder. The dataset is processed by the prepare_data file to obtain the LMDB database for accelerated training, and the database is stored in the mdb folder.The sample folder is the folder where the images are generated during the model training process, which can be used to traverse the training process. The generateSample folder is the sample image generated by calling StyleGenerator after the model training is completed.

The MultiResolutionDataset method for reading the LMDB database is defined in dataset.py, the Jittor model reproduced by Jittor is defined in model.py, train.py is used for the model training script, and VideoWrite.py is used to convert the generated image. output for video.

# Environment and execution instructions
Project environment dependencies include jittor, ldbm, PIL, argparse, tqdm and some common python libraries.

First you need to unzip the dataset in the data folder.
The model can be trained by the script in the terminal of the project environment
`python train.py --mixing "./mdb/color_symbol_7k_mdb"`

Images can be generated based on the trained model and compared for their differences by the script
`python generate.py --size 64 --n_row 3 --n_col 5 --path './checkpoint/040000.model'`

You can adjust the model training parameters by referring to the code in the args section of train.py and generate.py.

# Details



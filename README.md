# SalGAN

This is a clear implementation of the paper [SalGAN](https://arxiv.org/abs/1701.01081) in PyTorch. 

The  source  code  of  SalGAN  is  publicly  available<sup>[[1]](#SalGAN)</sup>,  but  it  is  written in Theano. There  is  also  a  PyTorch  implementation  of  SalGAN<sup>[[2]](#SalGAN_PyTorch)</sup>,  but many who tried to run the code report the mismatch of adversarial loss function between the original paper and their implementation. Here, in this implementation, the adversarial loss function is the same as stated in the original SalGAN paper.

The generative adversarial networks consist of two components:
- an encoder-decoder generator for saliency prediction and
- a discriminator for ground truth and generated saliency  distinction. 

The generator is a standard U-shape net, identical to the architecture of VGG-16 (encoder) followed by its reversed version (decoder).

The discrimnator is a smaller convolutional network with three fully connected layersattached to the end. The architecture of the system and the loss functions,and optimizers are all consistent with the SalGAN paper.

Note that the finalpooling  and  fully  connected  layers  is  removed  from  VGG16. 

---

If there is any problem with the code, please let me know or create an issue. 

---

<a name="SalGAN">1</a>: https://github.com/imatge-upc/salgan

<a name="SalGAN_PyTorch">2</a>: https://github.com/batsa003/salgan1

CSE 455

Name: Hugh Nguyen \
Project: Kaggle Bird Classification

1. Problem Description

To identify birds based on images using Deep Learning method.

2. Previous Work

Firstly, I used a resnet18 as the feature extractor with the pretrained model from pytorch's repository.
The result was decent training accuracy and validation accuracy.

3. My approach

After my initial test with the resnet18 in the guide, I decided to use deeper model.
I used resnet50 afterwards with some additional transformation.
After around 100 epochs with a learning rate of 1e-3, I achieved around ~90% training accuracy and ~80% validation accuracy. 
However, as I pushed the result onto Kaggle, I only received ~54% accuracy with test set.
I pushed deeper with resnet 152.
This time, I ran with ~100 epochs with bigger learning rate (0.01 same as the guide), lowering manually later to 1e-4
I ended up with a 99.7 training accuracy and 95% validation accuracy.
I pushed the result to Kaggle with a bit better result of ~66%.
Then, I pick the best performing saved epoch, 
and I changed the image size to 256 with much lower lr of 1e-5 with additional transformation of GaussianBlur only 
(on top of the original guide transformations).
This time, I achieved better result with ~72%

4. Dataset

Initially I split the kaggle training dataset into 2 subset with the ratio of 80:20 for training and validation set, respectively.
However, the test accuracy isn't as great as using the full dataset for training, so I turn off the validation code.

5. Results

The result is not as great as I expected, but I have tried my best to improve my result.

6. Discussion

- One of the problem for me is the lack of better model and the equipment to run it.
  Start off, I wanted to use Efficient Net but couldn't because I don't have sufficient hardware.
  So, I decided to use basic resnet from pytorch. Another problem is that I can't use GrayScale for this resnet.

- If I kept going with this project, I would like to use B7 efficient_net from Google with longer training time.
Also, I would be able to use more transformation approaches
  
- I added more transformations and tried to finetune learning rate manually based on loss and training accuracy. 

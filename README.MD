# SQUEEZE-RES

Convolutional Neural Networks (CNNs) are known to perform better with deeper architectures, as they can learn more complex features. However, deep networks often suffer from the vanishing gradient problem, where the gradients become extremely small during backpropagation, leading to slow or no weight updates and poor learning. This challenge has been addressed by using Residual Networks (ResNets), which introduce skip connections that allow gradients to bypass certain layers, mitigating the vanishing gradient issue. In this study, we propose a lightweight ResNet model that has less than 5 million trainable parameters as opposed to the original implementation of ResNet-18, which had around 11 million parameters. Through hyper-parameter tuning and data augmnetation techniques, our architecture achieves an accuracy of 92.17% when trained on the CIFAR-10 dataset. Squeeze-Res is a python implementation of different Res Architecture.

# Steps to train Resnet with Squeeze and exitement (92.17% and ~4.5M trainable parameters)
1. Open `driver.py` and check out the default args to the framework
2. Run command
```
python driver.py --pre_epoch=0 \
--epoch=100 \
--train_batch_size=128 \
--validation_batch_size=100 \
--lr=0.01 \
--momentum=0.9 \
--weight_decay=5e-4 \
--t_max=200 \
--img_channel=3 \
--num_classes=10 \
--optimizer='sgd' \
--model='lightnet' \
--scheduler=True \
--device="cuda"
```
>Note: model - lightnet is our implementation.
3. Once the training is done you can use the `run_details.json` to check out the various metrics like train loss, test loss, train acc, test acc.
4. `model.pth` is the final output model file. Load this model to play around further.

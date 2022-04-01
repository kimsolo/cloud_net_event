# cloud_net_event
For testing the event coding in point_cloud format.
为了方便阅读，采用中文readme

作为尝试，利用spiking_jelly平台结合点云网络尝试稀疏编码的event处理形式，目前已经跑通整体流程，感兴趣可以修改网络的编码形式。
目前输入是（n，3，2048）的伪点云形式，3表示(x, y, timestamp),2048表示event数量。

目前尝试了n-mnist数据集
网络选用了point-net的点云分类网络。

感兴趣可以提修改思路

envs:

spikingjelly == 0.0.0.0.8
tqdm
pillow
os

dataset:
from spikingjelly url:https://git.openi.org.cn/OpenI/spikingjelly/datasets?type=0
download to the corresponding path



import crypten
import torch
import crypten.mpc as mpc
import crypten.communicator as comm
from networks.u_net import Baseline
from torch.utils.data import DataLoader
from dataset import bladder
import os
import time

import utils.image_transforms as joint_transforms
import utils.transforms as extended_transforms
from utils.loss import *
from utils.metrics import diceCoeffv2

# para
crop_size = 256  # 输入裁剪大小
fold = 2
loss_name = 'dice'  # dice, bce, wbce, dual, wdual

# path
root_path = './'
model_path = os.path.join(root_path, "checkpoint/unet_bladder_maxpool_relu.pth")
val_path = os.path.join(root_path, 'media/Datasets/Bladder/raw_data')



def load_model():
    plaintext_model = Baseline()
    plaintext_model.load_state_dict(torch.load(model_path))
    # plaintext_model = torch.load("../checkpoint/bladder1.pth")
    # plaintext_model = crypten.load(model_path, dummy_model=dummy_model)
    # dummy_input = torch.empty(1, 1, 256, 256).cuda()
    dummy_input = torch.empty(1, 1, 256, 256)
    private_model = crypten.nn.from_pytorch(plaintext_model, dummy_input)
    private_model.encrypt()
    # private_model.cuda()
    crypten.print("Model successfully encrypted: ", private_model.encrypted)
    return private_model



def load_val_data(batch_size):
    # data preprocessing
    center_crop = joint_transforms.CenterCrop(crop_size)
    input_transform = extended_transforms.NpyToTensor()
    target_transform = extended_transforms.MaskToTensor()

    val_set = bladder.Dataset(val_path, 'val', fold,
                              joint_transform=None, transform=input_transform, center_crop=center_crop,
                              target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=1)
    return val_loader


def run_bladder_inference(learning_rate, batch_size, momentum):
    private_model = load_model()
    val_loader = load_val_data(batch_size)

    # setting
    criterion = None
    if loss_name == 'dice':
        criterion = SoftDiceLoss(bladder.num_classes)
    optimizer = crypten.optim.SGD(private_model.parameters(), lr=learning_rate, momentum=momentum)

    # inference
    private_model.eval()
    val_losses = []
    val_dice_arr = []
    val_batch = None
    val_class_dices = np.array([0] * (bladder.num_classes - 1), dtype=float)

    rank = comm.get().get_rank()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    ti = .0
    comms = .0

    private_model.to(device)

    for val_batch, ((input, mask), file_name) in enumerate(val_loader, 1):
        start_time = time.perf_counter()
        start_bytes = comm.get().get_communication_stats()['bytes']

        val_X = crypten.cryptensor(input, requires_grad=False)
        val_Y = mask

        val_X = val_X.to(device)
        val_Y = val_Y.to(device)

        pred = private_model(val_X)
        pred = pred.sigmoid()
        end_time = time.perf_counter()
        end_bytes = comm.get().get_communication_stats()['bytes']
        # crypten.print(f"Time {end_time-start_time:.4f}s  Comm {(end_bytes-start_bytes)/1024/1024:.4f}MB")
        ti += (end_time - start_time)
        comms += (end_bytes - start_bytes)

        pred = pred.get_plain_text()
        pred = pred.cpu().detach()
        val_Y = val_Y.cpu().detach()

        val_loss = criterion(pred, val_Y)
        val_losses.append(val_loss.item())

        # compute dice
        val_class_dice = []
        for i in range(1, bladder.num_classes):
            val_class_dice.append(diceCoeffv2(pred[:, i:i + 1, :], val_Y[:, i:i + 1, :]))
        # print("val_class_dice: ",val_class_dice)

        val_dice_arr.append(val_class_dice)
        val_class_dices = np.array(val_class_dice) + val_class_dices
        crypten.print("round {} : val_loss: {:.4}\t".format(val_batch, val_loss))

    val_loss = np.average(val_losses)
    val_class_dices = val_class_dices / val_batch
    val_mean_dice = val_class_dices.sum() / val_class_dices.size

    cc_ = comms / 1024 / 1024

    crypten.print(f"Inference Time: {ti:.8f} s")
    crypten.print(f"Inference Communication Cost: {cc_:.8f} MB total")

    crypten.print('val_loss: {:.4}\tval_mean_dice: {:.4}\tbladder: {:.4}\ttumor: {:.4}'
          .format(val_loss, val_mean_dice, val_class_dices[0], val_class_dices[1]))
    crypten.print('lr: {}'.format(optimizer.param_groups[0]['lr']))





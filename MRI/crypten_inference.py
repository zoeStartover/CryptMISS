import crypten
import torch
import time
import crypten.communicator as comm
import numpy as np
from model import UNetPP
from dataloader import get_loader
from utils import BCEDiceLoss, dice_coef

model_path = "./models/model_maxpool_relu.pth"


def load_model():
    plaintext_model = UNetPP(in_channels=3, num_classes=1, deep_supervision=False, init_features=32)
    plaintext_model.load_state_dict(torch.load(model_path))

    dummy_input = torch.empty(1, 3, 256, 256)
    private_model = crypten.nn.from_pytorch(plaintext_model, dummy_input)
    private_model.encrypt()
    crypten.print("Model successfully encrypted: ", private_model.encrypted)
    return private_model



def run_lgg_inference(batch_size):
    private_model = load_model()
    _, test_dl = get_loader(batch_size, num_workers=2)

    # setting
    criterion = BCEDiceLoss
    rank = comm.get().get_rank()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # inference
    private_model.to(device)
    private_model.eval()

    ti = .0
    comms = .0

    with crypten.no_grad():
        losses, nums, metrics = [], [], []
        test_num = len(test_dl)
        for i, (xb, yb_) in enumerate(test_dl):
            start_time = time.time()
            start_bytes = comm.get().get_communication_stats()['bytes']

            xb = (xb.to(device)).float()
            yb_ = (yb_.to(device)).float()
            yb = (yb_ > 0) * 1.0

            xb_crypten = crypten.cryptensor(xb, requires_grad=False)
            output = private_model(xb_crypten)

            end_time = time.time()
            end_bytes = comm.get().get_communication_stats()['bytes']
            crypten.print(f"Time {end_time - start_time:.4f}s  Comm {(end_bytes - start_bytes) / 1024 / 1024:.4f}MB")
            # crypten.print(output.get_plain_text(), yb)

            ti += end_time - start_time
            comms += end_bytes - start_bytes

            output = output.get_plain_text()
            loss = criterion(output, yb)
            loss = loss.cpu().detach().item()

            metric_score = dice_coef(output, yb)

            losses.append(loss)
            nums.append(xb.shape[0])
            metrics.append(metric_score)

            crypten.print(
                f"[{i}/{test_num}] Time:{end_time-start_time:.4f}s   Comm:{(end_bytes-start_bytes) / 1024 / 1024:.4f}MB  "
                f"Loss:{loss:.4f}   DICE:{metric_score:.4f}")

        losses = np.array(losses)
        metrics = np.array(metrics)
        nums = np.array(nums)

        total = np.sum(nums)
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        avg_metric = np.sum(np.multiply(metrics, nums)) / total

    crypten.print(f"Ciphertext [TEST:{total}] Time:{ti:.4f}s   Comm:{comms/1024/1024:.4f}MB  Loss:{avg_loss:.4f}   "
                  f"DICE:{avg_metric:.4f}")




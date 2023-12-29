import crypten
import torch
import time
from tqdm import tqdm
import crypten.communicator as comm

from unet import UNet
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from dice_loss import dice_coeff

# para
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# path
model_path = "./checkpoints/CP_epoch_maxpool_relu.pth"
dir_img = "./data/imgs"
dir_mask = "./data/masks"



def load_model(channels, classes):
    plaintext_model = UNet(n_channels=channels, n_classes=classes, bilinear=True).cuda()
    plaintext_model.load_state_dict(torch.load(model_path))
    dummy_input = torch.empty(1, 3, 256, 256).to(device)
    private_model = crypten.nn.from_pytorch(plaintext_model, dummy_input)
    private_model.encrypt()
    private_model.to(device)
    crypten.print("Model successfully encrypted: ", private_model.encrypted)
    return private_model



def load_val_data(batch_size=1, val_percent=0.1, img_scale=0.5):
    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    crypten.print("Loading data successfully...")
    return val_loader, n_train, n_val

def run_inference(learning_rate, batch_size, momentum, channels, classes, scale, val_percent):
    private_model = load_model(channels, classes)
    val_loader, n_train, n_val = load_val_data(batch_size, val_percent, scale)

    crypten.print(f'''Starting training:
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Device:          {device.type}
            Images scaling:  {scale}
        ''')

    optimizer = crypten.optim.SGD(private_model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=momentum)
    criterion = torch.nn.BCEWithLogitsLoss()

    # inference
    private_model.eval()

    mask_type = torch.float32
    tot = 0
    epoch_loss = 0.
    total_nums = 0
    times_ = 0.
    cc_ = 0.
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in val_loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            start_bytes = comm.get().get_communication_stats()['bytes']
            start_time = time.perf_counter()

            imgs = crypten.cryptensor(imgs)

            with torch.no_grad():
                mask_pred = private_model(imgs)

            end_time = time.perf_counter()
            end_bytes = comm.get().get_communication_stats()['bytes']
            times_ += (end_time - start_time)
            cc_ += (end_bytes - start_bytes)

            mask_pred = mask_pred.get_plain_text()
            loss = criterion(mask_pred, true_masks)
            epoch_loss += loss.item()

            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()

            tot += dice_coeff(pred, true_masks).item()
            total_nums += imgs.shape[0]
            temp_dice = tot / total_nums
            crypten.print(f"Loss: {loss:.4f}  Dice: {temp_dice:.4f}  Time: {end_time-start_time:.4f}s  Comm: {(end_bytes-start_bytes)/1024/1024:.4f}MB")
            pbar.update()

    dice = tot / max(n_val, 1)
    epoch_loss = epoch_loss / max(n_val, 1)

    cc_ = cc_ / 1024 / 1024

    crypten.print(f"Inference Time: {times_:.8f} s")
    crypten.print(f"Inference Communication Cost: {cc_:.8f} MB total")
    crypten.print('Validation Loss: {}  Dice Coeff: {}'.format(epoch_loss, dice))





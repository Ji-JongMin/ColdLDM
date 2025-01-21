import torch
from torch.optim import Adam
from tqdm.auto import tqdm
from accelerate import Accelerator
from utils import Dataset, cycle
from model import Autoencoder


class Trainer:
    def __init__(self, model, data_root, *, train_batch_size=32, train_lr=1e-4, train_num_steps=3000, amp=False):
        self.model = model
        self.train_num_steps = train_num_steps
        self.accelerator = Accelerator(mixed_precision='fp16' if amp else 'no')
        self.ds = Dataset(data_root)
        dl = torch.utils.data.DataLoader(self.ds, batch_size=train_batch_size, shuffle=True)
        self.dl = cycle(self.accelerator.prepare(dl))
        self.opt = Adam(self.model.parameters(), lr=train_lr)

    def train(self):
        self.model = self.accelerator.prepare(self.model)
        pbar = tqdm(total=self.train_num_steps, disable=not self.accelerator.is_main_process)
        for step in range(self.train_num_steps):
            img1, img2 = next(self.dl)
            img1, img2 = img1.cuda(), img2.cuda()
            with self.accelerator.autocast():
                output = self.model(img1)
                loss = torch.nn.functional.smooth_l1_loss(output, img2)
            self.accelerator.backward(loss)
            self.opt.step()
            self.opt.zero_grad()
            pbar.set_description(f"Step {step}/{self.train_num_steps} - Loss: {loss.item():.4f}")
            pbar.update(1)
        print("Training complete!")
      
dataroot = "path/to/your/data"
model = Autoencoder(input_ch=64, output_ch=64)
trainer = Trainer(model, dataroot, train_batch_size=32, train_lr=1e-4, train_num_steps=3000, amp=True)
trainer.train()      

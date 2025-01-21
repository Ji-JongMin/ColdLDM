from torch.optim import Adam
from accelerate import Accelerator
from tqdm.auto import tqdm
from diff_model import Unet_ddpm, Diffusion
from diff_sampling import DiffusionSampler
from torch.utils.data import DataLoader

class Trainer:
    def __init__(
        self,
        diffusion_model,
        ae,
        dataset,
        test_dataset,
        *,
        train_batch_size=4,
        test_batch_size=1,
        train_lr=1e-4,
        train_num_steps=100000
    ):
        self.accelerator = Accelerator()
        self.model = diffusion_model
        self.en = ae.encoder.eval()
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps

        self.sampler = DiffusionSampler(diffusion_model, ae.encoder, ae.decoder, diffusion_model.num_timesteps)

        self.dataloader = DataLoader(self.dataset, batch_size=train_batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=test_batch_size, shuffle=False)

        self.opt = Adam(self.model.parameters(), lr=train_lr)
        self.step = 0

    def train(self):
        device = self.accelerator.device
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self.dataloader = self.accelerator.prepare(self.dataloader)

        with tqdm(total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0

                for data100, data25 in self.dataloader:
                    data25 = data25.to(device).float()
                    data100 = data100.to(device).float()

                    cond = self.en(data25)
                    data100 = self.en(data100)
                    loss = self.model(data100, cond)

                    self.accelerator.backward(loss)
                    self.opt.step()
                    self.opt.zero_grad()

                    total_loss += loss.item()

                if self.step % 1000 == 0:
                    self.evaluate()

                pbar.set_description(f"Loss: {total_loss:.4f}")
                self.step += 1
                pbar.update(1)

    def evaluate(self):
        device = self.accelerator.device
        self.model.eval()

        with torch.no_grad():
            for img, _ in self.test_dataloader:
                img = img.to(device)
                output = self.sampler.sample(img)
                # Here you can save or visualize the output images
                print("Sample generated.")

        self.model.train()

import torch
import wandb
import time

# from drfp import DrfpEncoder
from torchmetrics import Accuracy

from visualise import plot_prediction, plot_reaction


class EnzymeGenerator: #TODO: to add wandb loging, saving and loading weights 
    def __init__(self, DENOISE, MAX_SEQUENCE_LENGTH, AMINO_ACID_TOKENSISER, N_STEPS=200, MODEL_PARAMS=None, WANDB=False, device=None, save_dir='weights/'):
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.N_STEPS = N_STEPS
        self.save_dir = save_dir
        self.device = device

        if WANDB:
            wandb.init(
            # set the wandb project where this run will be logged
            project="Theozmye-Transformer",
            
            # track hyperparameters and run metadata
            config=MODEL_PARAMS
        )
        self.wandb = WANDB

        self.aa_tokeniser = AMINO_ACID_TOKENSISER
        self.AA_size = len(self.aa_tokeniser)
        self.denoise = DENOISE

        self.accuracy = Accuracy(task='multiclass', num_classes=self.AA_size).to(self.device)
        
    def train(self, dataset, EPOCHS, BATCH_SIZE=5, learning_rate=1e-3, log=True, print_n=1 , name='enzyme', load=(True, True), visualiation=True):
        
        data_loader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)

        optimizer = torch.optim.Adam(self.denoise.model.parameters())

        optimizer = torch.optim.AdamW(self.denoise.model.parameters())
        schedular = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, epochs=EPOCHS)

        if load[0]:
            self.denoise.model.load_state_dict(torch.load(self.save_dir+name+'model.pt'))
        if load[1]:
            optimizer.load_state_dict(torch.load(self.save_dir+name+'optimizer-adam.pt'))

        n_batches = len(data_loader)
        start = time.time()

        for epoch in range(EPOCHS):
            tot_loss = 0
            for AA_seq, eqx in data_loader:
                AA_seq = torch.where(AA_seq > 0, AA_seq, 0)
                AA_seq = torch.nn.functional.one_hot(torch.round(AA_seq).long(), self.AA_size).float()
                loss = self.denoise.loss(AA_seq, eqx[:, :, None])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tot_loss += loss.detach().item()

            torch.save(self.denoise.model.state_dict(), self.save_dir+name+'model.pt')
            torch.save(optimizer.state_dict(), self.save_dir+name+'optimizer-adamW.pt')
            
            schedular.step()

            if epoch % print_n == 0:
                l = tot_loss / n_batches
                xt = self.inference(eqx[:, :, None].to(self.device), 2)
                true_seq = self.aa_tokeniser(AA_seq.detach()[-1].argmax(-1))
                pred_seq = self.aa_tokeniser(xt.detach()[-1].argmax(-1))

                acc = self.accuracy(torch.einsum('ijk->ikj', xt), AA_seq.argmax(-1))
                print(f"{time.ctime(time.time())} | epoch : {epoch} | loss : {str(l)} | acc : {str(acc )} | {round((time.time()-start)/60, 3)}mins") 
                
                log = {
                        "epoch" : epoch,
                        "loss" : l,
                        "acc" : acc,
                        "true-seq" : true_seq,
                        "pred-seq" : pred_seq
                    }

                if visualiation:
                    conf_img, seq_img = plot_prediction(xt.detach()[-1].cpu().float(), AA_seq.detach()[-1].cpu().float()) #take last sample
                    log['conf-img'] = wandb.Image(conf_img)
                    log['seq-img'] = wandb.Image(seq_img)

                    # reaction = plot_reaction(details[-1][1])


                if self.wandb:
                    wandb.log(log)
                start = time.time()
                


    def inference(self, cond, batch_size=5):
        batch_size = cond.shape[0]
        with torch.no_grad():
            xt = torch.rand(
            (batch_size, self.MAX_SEQUENCE_LENGTH, self.AA_size), device=self.device
            )    
            ts = torch.arange(self.N_STEPS-1, 1, -1, device=self.device)
            for t in ts:
                noise = self.denoise.model.forward(xt, torch.tensor([t], device=self.device), cond)
                xt = (1/self.denoise.alpha[t]**.5)*(xt - ((1 - self.denoise.alpha[t])/(1-self.denoise.alpha_bar[t])**.5)*noise) + self.denoise.sigma2[t]*torch.rand_like(xt)
            
            return xt

    def predict(self, cond, batch_size=5):
        xt = self.inference(cond, batch_size)
        xt = xt.argmax(-1)
        xt = [self.aa_tokeniser(a) for a in xt]
        return xt
import torch
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# from train import dataset

def look_ahead_mask(shape):
  mask = torch.arange(shape)[None, :] > torch.arange(shape)[:, None]
  return mask

def scaled_product_attention(q, k, v, padding_mask = None):
  x = torch.matmul(q, torch.einsum('ijk->ikj', k))
  x = x/k.shape[1]**0.5

  x = torch.softmax(x, dim=2)
  x = torch.einsum('bij,bid->bid', x, v)
  return x

def scaled_product_attention_4D(q, k, v, weights=None, mask=None, padding_mask=None, look_ahead_mask=None):
  x = torch.matmul(q, torch.einsum('ijkm->ijmk', k))
  x = x/k.shape[1]**0.5

  if weights is not None:
    x = (x[:, :, :, None] * weights).sum(-1)

  if look_ahead_mask is not None:
    x[:, look_ahead_mask] = float('-inf')      
  
  if padding_mask is not None:
    x[padding_mask] = float('-inf')

  x = torch.softmax(x, dim=2)
  x = torch.matmul(x, v)
  return x

class MHA(torch.nn.Module):
  def __init__(self, dmodel, nheads, attn=scaled_product_attention):
    super().__init__()
    self.dmodel = dmodel
    self.nheads = nheads
    self.attn = attn

    
    self.wq = torch.nn.ModuleList([torch.nn.Linear(self.dmodel, self.dmodel) for i in range(nheads)]).to(device)
    self.wk = torch.nn.ModuleList([torch.nn.Linear(self.dmodel, self.dmodel) for i in range(nheads)]).to(device)
    self.wv = torch.nn.ModuleList([torch.nn.Linear(self.dmodel, self.dmodel) for i in range(nheads)]).to(device)

    self.out = torch.nn.Linear(self.dmodel*nheads, self.dmodel).to(device)


  def forward(self, Q, K, V, mask=None):
    qkv = []
    for head in range(self.nheads):
      wq = self.wq[head](Q)
      wk = self.wk[head](K)
      wv = self.wv[head](V) 
      attn = self.attn(wq, wk, wv, mask)
  
      qkv.append(attn)
    qkv = torch.concat(qkv, axis=-1).to(device)
    qkv = self.out(qkv)
    return qkv


class encoder_layer(torch.nn.Module):
  def __init__(self, dmodel, nheads, dff, dropout=0.001):
    super().__init__()

    self.attn = torch.nn.MultiheadAttention(dmodel, nheads, dropout=dropout, batch_first=True)
    self.feadforward = torch.nn.Sequential(
        torch.nn.Linear(dmodel, dff),
        torch.nn.Dropout(dropout),
        torch.nn.ReLU(),
        torch.nn.Linear(dff, dmodel),
        torch.nn.Dropout(dropout),
    ).to(device)

    self.norm1 = torch.nn.LayerNorm(dmodel).to(device)
    self.norm2 = torch.nn.LayerNorm(dmodel).to(device)

  def forward(self, x):
    attn, _ = self.attn(x, x, x)
    x = self.norm1(attn+x)
    ff = self.feadforward(x)
    x = self.norm2(ff+x)
    return x

class decoder_layer(torch.nn.Module):   
  def __init__(self, dmodel, nheads, dff, dropout=0.001):
    super().__init__()

    self.attn = torch.nn.MultiheadAttention(dmodel, nheads, dropout=dropout, batch_first=True)
    self.feadforward = torch.nn.Sequential(
        torch.nn.Linear(dmodel, dff),
        torch.nn.Dropout(dropout),
        torch.nn.ReLU(),
        torch.nn.Linear(dff, dmodel),
        torch.nn.Dropout(dropout),
    ).to(device)
    self.norm1 = torch.nn.LayerNorm(dmodel).to(device)
    self.norm2 = torch.nn.LayerNorm(dmodel).to(device)
    self.norm3 = torch.nn.LayerNorm(dmodel).to(device)

    self.cross_attention = torch.nn.MultiheadAttention(dmodel, nheads, dropout=dropout, batch_first=True)


  def forward(self, x, latent_x):

    attn, _ = self.attn(x, x, x)
    x = self.norm1(attn+x)

    attn, _ = self.cross_attention(x, latent_x, latent_x)
    x = self.norm2(attn+x)

    ff = self.feadforward(x)
    x = self.norm3(ff+x)

    return x
    

class Transformer(torch.nn.Module):
    def __init__(self, sequence_size=1000, chem_size=256, dropout=0.1, dmodel=20, dff=200, nheads=4, nblocks=4, AA_size=21, MAX_TIME=200, Encode_cond=True):   
        super().__init__()
        self.sequence_size = sequence_size
        self.chem_size = chem_size
        self.aa_size = AA_size
        self.dff = dff
        self.dmodel = dmodel
        self.nheads = nheads
        self.nblocks = nblocks
        self.max_time = MAX_TIME

        self.position_embed = positionalencoding1d(self.dmodel, sequence_size).to(device)

        self.embed_inp = torch.nn.Sequential(torch.nn.Linear(self.aa_size, self.dmodel))
        self.embed_cond = torch.nn.Sequential(torch.nn.Linear(1, self.dmodel))

        if Encode_cond:
          self.encoder = torch.nn.ModuleList([encoder_layer(dmodel, nheads, dff, dropout) for i in range(nblocks)])
        else:
          self.encoder = torch.nn.ModuleList([])

        self.decoder = torch.nn.ModuleList([decoder_layer(dmodel, nheads, dff, dropout) for i in range(nblocks)])

        self.output = torch.nn.Sequential(
          torch.nn.Linear(dmodel, dff),
          torch.nn.ReLU(),
          torch.nn.Linear(dff, AA_size),
        )


    def forward(self, xt, t, cond):
        xt = self.embed_inp(xt)
        xt += self.position_embed[None, :, :]
        xt += time_step_embedding(t, dmodel=self.dmodel, max_period=self.max_time)[:, None, :]
        
        cond = self.embed_cond(cond)

        for encoder in self.encoder:
          cond = encoder(cond)

        for decoder in self.decoder:
          xt = decoder(xt, cond)

        xt = self.output(xt)
        return xt

    # def cond_foward(self, cond):
    #   cond = self.embed_cond(cond)
    #   for encoder in self.encoder:
    #     cond = encoder(cond)
    #   return cond

    # def inference(self, cond, T):
    #   if len(cond.size) > 2:
    #     batch_size = cond.shape[0]
    #   else:
    #     batch_size = 1
    #     cond = cond.reshape((batch_size, cond.shape[0], cond.shape[1]))

    #   with torch.no_grad():
    #     xt = torch.randn(
    #       (batch_size, self.sequence_size, self.aa_size), device=self.device)
    #       ts = torch.arange(T-1, 1, -1, device=self.device)
    #       cond = self.cond_foward(cond)
    #       for t in ts:



class temporal_embed():
  def __init__(self, T=1000) -> None:
    self.pe = torch.zeros(T)
    position = torch.arange(0, T).unsqueeze(1)
    div_term = torch.exp(
      (torch.arange(0, T, dtype=torch.float) * -(math.log(10000) / T)))
    self.pe[0::2] = torch.sin(position.float()*div_term)
    self.pe[1::2] = torch.cos(position.float()*div_term)
  
  def __call__(self, t):
    x = torch.gather(self.pe, t)
    return x

class SinusoidalPositionEmbeddings(torch.nn.Module):
    def __init__(self, max_T):
        super().__init__()
        half_dim = max_T // 2
        self.div_term = math.log(10000) / (half_dim - 1)

    def forward(self, time):
        evens = torch.where(time % 2 == 0, time, 0)
        odds = torch.where(time % 2 == 1, time, 0)
        
        evens = torch.exp(evens*-self.div_term).sin()
        odds = torch.exp(odds*-self.div_term).cos()
        return evens + odds

def positionalencoding1d(d_model, length): #from https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

def time_step_embedding(time_steps: torch.Tensor, max_period: int = 10000, dmodel=24):
    """
    ## Create sinusoidal time step embeddings
    :param time_steps: are the time steps of shape `[batch_size]`
    :param max_period: controls the minimum frequency of the embeddings.
    """
    half = dmodel // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=device) / half
    )
    args = time_steps[:, None].float() * frequencies[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

def get_padding(element): #arg element : (Batch size, sequence length, int)
  lengths = element.argmin(1)#gets length of nonpadded array, as it returns the index of the first zero
  mask = torch.arange(element.shape[1]).to(device)[None, :] > lengths[:, None]
  mask = mask[:, :, None].repeat(1, 1, mask.shape[1])
  
  return mask


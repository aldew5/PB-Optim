import torch
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# params for PAC-Bayes bound optimization (see Roy et al. 2017)
delta = torch.tensor(0.025, dtype=torch.float32).to(device)
m = torch.tensor(60000, dtype=torch.float32).to(device)
b = torch.tensor(100, dtype=torch.float32).to(device)
#c = torch.tensor(.1, dtype=torch.float32).to(device)
c = torch.tensor(.1, dtype=torch.float32).to(device)
pi = torch.tensor(math.pi, dtype=torch.float32).to(device)
delta_prime = torch.tensor(0.01, dtype=torch.float32).to(device)


#roughly sigma^2 = 0.05
p_log_sigma = torch.tensor(-1.5, dtype=torch.float32).to(device)
gamma_ex= torch.tensor(1e-6, dtype=torch.float32).to(device)

SAVE_WEIGHTS = True
LOAD_DATA = False
'''
Author @ 2024 Dongyang Kuang

Containing network modules
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Backbone(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Backbone, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Backbone_sep(nn.Module):
    '''
    Assuming the pressure and energy are of separable form
        P = P(V,T) = f(V)g(T) + h(V)
        E = E(V,P) = f(V)g(P) + h(V)
    '''
    def __init__(self, hidden_size, droput_rate=0.0, last_activation='linear'):
        super(Backbone_sep, self).__init__()
        self.coef_P_fv = self.build_mini_net(hidden_size, droput_rate)
        self.coef_P_hv = self.build_mini_net(hidden_size, droput_rate)
        self.coef_P_gT = self.build_mini_net(hidden_size, droput_rate)
        self.last_activation = last_activation
    def build_mini_net(self,  hidden_size, droput_rate):
        return nn.Sequential(
                            nn.Linear(1, hidden_size),
                            # nn.ReLU(),
                            nn.Tanh(),
                            # nn.Tanh(),
                            nn.Dropout(droput_rate),
                            nn.Linear(hidden_size, 1)
        )


    def forward(self, x):
        fv = self.coef_P_fv(x[:, :1])
        gT = self.coef_P_gT(x[:, 1:])
        hv = self.coef_P_hv(x[:, :1])
        if self.last_activation == 'sigmoid':
            # return nn.Sigmoid()(fv*gT + hv)
            return nn.functional.sigmoid(fv*gT + hv)
        elif self.last_activation == 'relu':
            # return nn.Sigmoid()(fv*gT + hv)
            return nn.functional.relu(fv*gT + hv)
        else:
            return fv*gT + hv 
    
class Backbone_sep_V1(nn.Module):
    '''
    Assuming the pressure and energy are of separable form
        P = P(V,T) = f(V)g(T) + h(V)
        E = E(V,P) = f(V)g(P) + h(V)
    a dropout-linear layer is added
    '''
    def __init__(self, hidden_size, dropout_rate=0.0, feature_dim=2):
        super(Backbone_sep_V1, self).__init__()
        self.feature_dim = feature_dim

        self.coef_P_fv = self.build_mini_net(hidden_size)
        self.coef_P_hv = self.build_mini_net(hidden_size)
        self.coef_P_gT = self.build_mini_net(hidden_size)
                
        self.linear = nn.Linear(self.feature_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def build_mini_net(self,  hidden_size):
        return nn.Sequential(
                            nn.Linear(1, hidden_size),
                            # nn.ReLU(),
                            nn.Tanh(),
                            nn.Linear(hidden_size, self.feature_dim)
        )


    def forward(self, x):
        fv = self.coef_P_fv(x[:, :1])
        gT = self.coef_P_gT(x[:, 1:])
        hv = self.coef_P_hv(x[:, :1])
        
        out = self.dropout(fv*gT + hv)
        out = self.linear(out)

        return out
    
class Backbone_sep_V2(nn.Module):
    '''
    Assuming the pressure and energy are of separable form
        P = P(V,T) = f(V)g(T) + h(V)
        E = E(V,P) = f(V)g(P) + h(V)
    '''
    def __init__(self, hidden_size, droput_rate=0.0, last_activation='linear'):
        super(Backbone_sep_V2, self).__init__()
        self.coef_P_fv = self.build_mini_net(hidden_size, droput_rate)
        self.coef_P_hv = self.build_mini_net(hidden_size, droput_rate)
        self.coef_P_gT = self.build_mini_net(hidden_size, droput_rate)
        self.coef_P_kT = self.build_mini_net(hidden_size, droput_rate)
        
        self.last_activation = last_activation
    def build_mini_net(self,  hidden_size, droput_rate):
        return nn.Sequential(
                            nn.Linear(1, hidden_size),
                            # nn.ReLU(),
                            nn.Tanh(),
                            # nn.Tanh(),
                            nn.Dropout(droput_rate),
                            nn.Linear(hidden_size, 1)
        )


    def forward(self, x):
        fv = self.coef_P_fv(x[:, :1])
        gT = self.coef_P_gT(x[:, 1:])
        hv = self.coef_P_hv(x[:, :1])
        kT = self.coef_P_kT(x[:, 1:])
        if self.last_activation == 'sigmoid':
            # return nn.Sigmoid()(fv*gT + hv)
            return nn.functional.sigmoid(fv*gT + hv + kT)
        elif self.last_activation == 'relu':
            # return nn.Sigmoid()(fv*gT + hv)
            return nn.functional.relu(fv*gT + hv + kT)
        else:
            return fv*gT + hv + kT

class Backbone_sep_V3(nn.Module):
    '''
    Assuming the pressure and energy are of separable form
        P = P(V,T) = f(V)g(T) + h(V)
        E = E(V,P) = f(V)g(P) + h(V)
    '''
    def __init__(self, hidden_size, droput_rate=0.0, last_activation='linear'):
        super(Backbone_sep_V3, self).__init__()
        self.coef_P_fv = self.build_mini_net(hidden_size, droput_rate)
        self.coef_P_hv = self.build_mini_net(hidden_size, droput_rate)
        self.coef_P_gT = self.build_mini_net(hidden_size, droput_rate)
        self.last_activation = last_activation
    def build_mini_net(self,  hidden_size, droput_rate):
        return nn.Sequential(
                            nn.Linear(1, hidden_size),
                            # nn.ReLU(),
                            nn.Tanh(),
                            nn.Linear(hidden_size, hidden_size//2),
                            nn.ReLU(),
                            # nn.Tanh(),
                            nn.Dropout(droput_rate),
                            nn.Linear(hidden_size//2, 1)
        )


    def forward(self, x):
        fv = self.coef_P_fv(x[:, :1])
        gT = self.coef_P_gT(x[:, 1:])
        hv = self.coef_P_hv(x[:, :1])
        if self.last_activation == 'sigmoid':
            # return nn.Sigmoid()(fv*gT + hv)
            return nn.functional.sigmoid(fv*gT + hv)
        elif self.last_activation == 'relu':
            # return nn.Sigmoid()(fv*gT + hv)
            return nn.functional.relu(fv*gT + hv)
        else:
            return fv*gT + hv

class Backbone_sep_prob(nn.Module):
    '''
    Assuming the pressure and energy are of separable form
        P = P(V,T) = f(V)g(T) + h(V)
        E = E(P,V) = f(P)g(V) + h(P)

    a probability version that outputs a normal distribution
    '''
    def __init__(self, hidden_size, dropout_rate=0.0):
        super(Backbone_sep_prob, self).__init__()
        self.dropout_rate = dropout_rate
        self.coef_P_fv = self.build_mini_net(hidden_size)
        self.coef_P_hv = self.build_mini_net(hidden_size)
        self.coef_P_gT = self.build_mini_net(hidden_size)

        self.std_net = self.build_var_net(hidden_size)
        

    def build_mini_net(self,  hidden_size):
        return nn.Sequential(
                            nn.Linear(1, hidden_size),
                            # nn.ReLU(),
                            nn.Dropout(self.dropout_rate),
                            nn.Tanh(),
                            nn.Linear(hidden_size, 1)
        )

    def build_var_net(self, hidden_size):
        return nn.Sequential(
            nn.Linear(2, hidden_size),
            # nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softplus(),  # enforces positivity
        )

    def forward(self, x):
        fv = self.coef_P_fv(x[:, :1])
        gT = self.coef_P_gT(x[:, 1:])
        hv = self.coef_P_hv(x[:, :1])

        std = self.std_net(x)
        return torch.distributions.Normal(fv*gT + hv, std+1e-6)

    
class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out += residual
        out = self.relu(out)
        return out

class ResidualNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_blocks, output_size):
        super(ResidualNet, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.residual_blocks = nn.ModuleList([ResidualBlock(hidden_size, hidden_size) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.linear(x)
        out = self.relu(out)
        for block in self.residual_blocks:
            out = block(out)
        out = self.output_layer(out)
        return out

class latent_basis_model(nn.Module):
    '''
    y = Poly(w,x) where x is the latent variable, w depends on all training data
    '''
    def __init__(self, Xsize, latent_dim=4, Xlim=2, concateX=True, 
                 hidden_size=16,num_blocks=2, l2_reg=0.0, **kwargs):
        super(latent_basis_model, self).__init__(**kwargs)
        # self.model = model
        self.model = ResidualNet(input_size=Xlim, 
                               output_size=latent_dim,
                               hidden_size=hidden_size, 
                               num_blocks=num_blocks)
        self.Xlim = Xlim
        self.concateX = concateX
        if concateX:
            self.linear = nn.Linear(Xsize+latent_dim, 1)
        else:
            self.linear = nn.Linear(latent_dim, 1)
        self.l2_reg = l2_reg
    
    def forward(self, input):
        latent_x = self.model(input[:, :self.Xlim])

        if self.concateX:
            latent_x = torch.cat([input, latent_x], dim=-1)
        pred = self.linear(latent_x)
        return pred
    def get_loss(self, input, target):
        pred = self.forward(input)
        loss = F.mse_loss(pred, target)
        l2_reg = 0
        for param in self.linear.parameters():
            l2_reg += torch.norm(param)
        loss += self.l2_reg * l2_reg
        return loss
    
class Joint_net(nn.Module):
    def __init__(self, pnet, enet):
        super(Joint_net, self).__init__()
        self.pnet = pnet
        self.enet = enet


    def forward(self, x):
        # v, T = torch.split(x, 1, dim=1)
        p = self.pnet(x)  # p = p(V,T)
        vp = torch.cat((x[:,:1],p), dim=-1) 
        e = self.enet(vp) # e = e(p,V)
      
        return e, p
    
class Joint_net_prob(nn.Module):
    def __init__(self, pnet, enet):
        super(Joint_net_prob, self).__init__()
        self.pnet = pnet
        self.enet = enet


    def forward(self, x):
        # v, T = torch.split(x, 1, dim=1)
        p = self.pnet(x)  # p = p(V,T)
        p_mean = p.mean
        vp = torch.cat((x[:,:1],p_mean), dim=-1) 
        e = self.enet(vp) # e = e(p,V)
      
        return e, p

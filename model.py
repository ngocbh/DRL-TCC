import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.embed = nn.Linear(input_size, hidden_size)
        self.embed_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, input):
        """forward.

        Parameters
        ----------
        input : (batch_size, *, input_size)
            input

        Returns:
        ----------
        output : (batch_size, *, hidden_size)
        """
        output = F.relu(self.embed(input))
        output = self.embed_2(output)
        return output 

class Attention(nn.Module):
    def __init__(self, hidden_size, device='cpu'):
        super(Attention, self).__init__()

        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size),
                                          device=device, requires_grad=True))

    def forward(self, mc_hidden, sn_hidden):
        batch_size, hidden_size, _ = sn_hidden.size()

        hidden = mc_hidden.unsqueeze(2).expand_as(sn_hidden)
        hidden = torch.cat((sn_hidden, hidden), 1)

        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = F.softmax(attns, dim=2)
        return attns

class Pointer(nn.Module):
    def __init__(self, hidden_size, device='cpu'):
        super(Pointer, self).__init__()

        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.encoder_attn = Attention(hidden_size, device)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, mc_hidden, sn_hidden):
        """forward.

        Parameters
        ----------
        mc_hidden : (batch_size, hidden_size)
            mc_hidden
        sn_hidden : (batch_size, hidden_size, num_sensors)
            sn_hidden
        """
        enc_attn = self.encoder_attn(mc_hidden, sn_hidden) # (batch_size, 1, num_sensors)
        context = enc_attn.bmm(sn_hidden.permute(0, 2, 1)) # (batch_size, 1, hidden_size)

        # (batch_size, hidden_size * 2)
        input = torch.cat((mc_hidden, context.squeeze(1)), dim=1)

        output = F.relu(self.fc1(input))
        output = F.relu(self.fc2(output)) # (batch_size, hidden_size)

        output = output.unsqueeze(2).expand_as(sn_hidden)

        v = self.v.expand(sn_hidden.size(0), -1, -1) # (batch_size, 1, hidden_size)
        probs = torch.bmm(v, torch.tanh(sn_hidden + output)).squeeze(1)
        # probs = F.softmax(probs, dim=1)

        return probs

class MCActor(nn.Module):
    def __init__(self, mc_input_size, sn_input_size,
                 hidden_size, dropout=0., device='cpu'):
        super(MCActor, self).__init__()

        # Define the encoder & decoder models
        self.mc_encoder = Encoder(mc_input_size, hidden_size)
        self.sn_encoder = Encoder(sn_input_size, hidden_size)
        self.pointer = Pointer(hidden_size, device=device)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, mc_input, sn_input):
        """forward.

        Parameters
        ----------
        mc_input : (batch_size, mc_input_size)
            mc_input
        sn_input : (batch_size, num_sensors, sn_input_size)
            sn_input
        """
        mc_hidden = self.mc_encoder(mc_input)
        sn_hidden = self.sn_encoder(sn_input)
        probs = self.pointer(mc_hidden, sn_hidden.permute(0, 2, 1))
        return probs

class Critic(nn.Module):
    def __init__(self, mc_input_size, sn_input_size, hidden_size):
        super(Critic, self).__init__()

        self.mc_encoder = Encoder(mc_input_size, hidden_size)
        self.sn_encoder = Encoder(sn_input_size, hidden_size)

        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 20)
        self.fc3 = nn.Linear(20, 1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, mc_input, sn_input):
        mc_hidden = self.mc_encoder(mc_input)
        sn_hidden = self.sn_encoder(sn_input)

        hidden = mc_hidden.unsqueeze(1).expand_as(sn_hidden)
        hidden = torch.cat((hidden, sn_hidden), 2)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=1)
        return output



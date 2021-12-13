import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from models.pse import PixelSetEncoder
from models.tae import TemporalAttentionEncoder
from models.ltae import LTAE
from models.decoder import get_decoder
from models.competings import GRU, TempConv


class PseLTaeCombinedPlanetS1S2(nn.Module):
    """
    Pixel-Set encoder + Lightweight Temporal Attention Encoder sequence classifier
    """

    def __init__(self, input_dim_planet=5, input_dim_s1=3, input_dim_s2=13, mlp1_planet=[5, 32, 64], mlp1_s1=[3, 32, 64], mlp1_s2=[13, 32, 64], pooling='mean_std', mlp2=[132, 128],
                 with_extra=True, extra_size=4, n_head=16, d_k=8, d_model=256, mlp3_planet=[256, 128], mlp3_s1=[256, 64], mlp3_s2=[256, 64], dropout=0.2, T=1000,
                 len_max_seq_planet=244, len_max_seq_s1=41, len_max_seq_s2=76, positions=None, mlp4=[128+64, 64, 32, 20], return_att=False):
        super(PseLTaeCombinedPlanetS1S2, self).__init__()
        
        # if extras is true then include it only in planet model
        self.spatial_encoder_planet = PixelSetEncoder(input_dim_planet, mlp1=mlp1_planet, pooling=pooling, mlp2=mlp2, with_extra=with_extra,
                                               extra_size=extra_size)
        self.temporal_encoder_planet = LTAE(in_channels=mlp2[-1], n_head=n_head, d_k=d_k,
                                           d_model=d_model, n_neurons=mlp3_planet, dropout=dropout,
                                           T=T, len_max_seq=len_max_seq_planet, positions=positions, return_att=return_att
                                           )
        
        if with_extra: mlp2[0] = mlp2[0] - extra_size
            
        self.spatial_encoder_s1 = PixelSetEncoder(input_dim_s1, mlp1=mlp1_s1, pooling=pooling, mlp2=mlp2, with_extra=False,
                                               extra_size=extra_size)
        self.temporal_encoder_s1 = LTAE(in_channels=mlp2[-1], n_head=n_head, d_k=d_k,
                                           d_model=d_model, n_neurons=mlp3_s1, dropout=dropout,
                                           T=T, len_max_seq=len_max_seq_s1, positions=positions, return_att=return_att
                                           )
        self.spatial_encoder_s2 = PixelSetEncoder(input_dim_s2, mlp1=mlp1_s2, pooling=pooling, mlp2=mlp2, with_extra=False,
                                               extra_size=extra_size)
        self.temporal_encoder_s2 = LTAE(in_channels=mlp2[-1], n_head=n_head, d_k=d_k,
                                           d_model=d_model, n_neurons=mlp3_s2, dropout=dropout,
                                           T=T, len_max_seq=len_max_seq_s2, positions=positions, return_att=return_att
                                           )
        self.decoder = get_decoder(mlp4)
        self.return_att = return_att

    def forward(self, input):
        """
         Args:
            input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
            Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
            Pixel-Mask : Batch_size x Sequence length x Number of pixels
            Extra-features : Batch_size x Sequence length x Number of features
        """
        
        input1, input2, input3 = input
        out1 = self.spatial_encoder_planet(input1) # out size is 8,48,128
        out2 = self.spatial_encoder_s1(input2)
        out3 = self.spatial_encoder_s2(input3)
        
        if self.return_att:
            out1, att1 = self.temporal_encoder_planet(out1)
            out2, att2 = self.temporal_encoder_s1(out2)
            out3, att3 = self.temporal_encoder_s2(out3)
            out = torch.cat([out1, out2, out3], dim=1)
            out = self.decoder(out)
            return out, (att1, att2)
        else:
            out1 = self.temporal_encoder_planet(out1)
            out2 = self.temporal_encoder_s1(out2)
            out3 = self.temporal_encoder_s2(out3)
            out = torch.cat([out1, out2, out3], dim=1)            
            out = self.decoder(out)
            return out

    def param_ratio(self):
        total = get_ntrainparams(self)
        s = get_ntrainparams(self.spatial_encoder)
        t = get_ntrainparams(self.temporal_encoder)
        c = get_ntrainparams(self.decoder)

        print('TOTAL TRAINABLE PARAMETERS : {}'.format(total))
        print('RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%'.format(s / total * 100,
                                                                                          t / total * 100,
                                                                                          c / total * 100))

        return total


    
class PseLTaeCombinedPlanetS1(nn.Module):
    """
    Pixel-Set encoder + Lightweight Temporal Attention Encoder sequence classifier
    """

    def __init__(self, input_dim_planet=5, input_dim_s1=3, mlp1_planet=[5, 32, 64], mlp1_s1=[3, 32, 64], pooling='mean_std', mlp2=[132, 128], with_extra=True, 
                 extra_size=4, n_head=16, d_k=8, d_model=256, mlp3_planet=[256, 128], mlp3_s1=[256, 64],dropout=0.2, T=1000, len_max_seq_planet=244, len_max_seq_s1=41,positions=None,
                 mlp4=[128+64, 64, 32, 20], return_att=False):
        super(PseLTaeCombinedPlanetS1, self).__init__()
        
        # if extras is true then include it only in planet model
        self.spatial_encoder_planet = PixelSetEncoder(input_dim_planet, mlp1=mlp1_planet, pooling=pooling, mlp2=mlp2, with_extra=with_extra,
                                               extra_size=extra_size)
        self.temporal_encoder_planet = LTAE(in_channels=mlp2[-1], n_head=n_head, d_k=d_k,
                                           d_model=d_model, n_neurons=mlp3_planet, dropout=dropout,
                                           T=T, len_max_seq=len_max_seq_planet, positions=positions, return_att=return_att
                                           )
        
        if with_extra: mlp2[0] = mlp2[0] - extra_size
        
        self.spatial_encoder_s1 = PixelSetEncoder(input_dim_s1, mlp1=mlp1_s1, pooling=pooling, mlp2=mlp2, with_extra=False,
                                               extra_size=extra_size)
        self.temporal_encoder_s1 = LTAE(in_channels=mlp2[-1], n_head=n_head, d_k=d_k,
                                           d_model=d_model, n_neurons=mlp3_s1, dropout=dropout,
                                           T=T, len_max_seq=len_max_seq_s1, positions=positions, return_att=return_att
                                           )
        self.decoder = get_decoder(mlp4)
        self.return_att = return_att

    def forward(self, input):
        """
         Args:
            input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
            Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
            Pixel-Mask : Batch_size x Sequence length x Number of pixels
            Extra-features : Batch_size x Sequence length x Number of features
        """
        
        input1, input2 = input
        
        out1 = self.spatial_encoder_planet(input1) # out size is 8,48,128
        out2 = self.spatial_encoder_s1(input2)
        
        if self.return_att:
            out1, att1 = self.temporal_encoder_planet(out1)
            out2, att2 = self.temporal_encoder_s1(out2)
            out = torch.cat([out1, out2], dim=1)
            out = self.decoder(out)
            return out, (att1, att2)
        else:
            out1 = self.temporal_encoder_planet(out1)
            out2 = self.temporal_encoder_s1(out2)
            #print('aaaaaaaaaaaaa',out1.size(), out2.size())
            out = torch.cat([out1, out2], dim=1)   
            out = self.decoder(out)
            return out

    def param_ratio(self):
        total = get_ntrainparams(self)
        s = get_ntrainparams(self.spatial_encoder)
        t = get_ntrainparams(self.temporal_encoder)
        c = get_ntrainparams(self.decoder)

        print('TOTAL TRAINABLE PARAMETERS : {}'.format(total))
        print('RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%'.format(s / total * 100,
                                                                                          t / total * 100,
                                                                                          c / total * 100))

        return total




class PseLTae_pretrained(nn.Module):

    def __init__(self, weight_folder, hyperparameters, device='cuda', fold='all'):
        """
        Pretrained PseLTea classifier.
        The class can either load the weights of a single fold or aggregate the predictions of the different sets of
        weights obtained during k-fold cross-validation and produces a single prediction.
        Args:
            weight_folder (str): Path to the folder containing the different sets of weights obtained during each fold
            (res_dir of the training script)
            hyperparameters (dict): Hyperparameters of the PseLTae classifier
            device (str): Device on which the model should be loaded ('cpu' or 'cuda')
            fold( str or int): load all folds ('all') or number of the fold to load
        """
        super(PseLTae_pretrained, self).__init__()
        self.weight_folder = weight_folder
        self.hyperparameters = hyperparameters

        self.fold_folders = [os.path.join(weight_folder, f) for f in os.listdir(weight_folder) if
                             os.path.isdir(os.path.join(weight_folder, f))]
        if fold == 'all':
            self.n_folds = len(self.fold_folders)
        else:
            self.n_folds = 1
            self.fold_folders = [self.fold_folders[int(fold) - 1]]
        self.model_instances = []

        print('Loading pre-trained models . . .')
        for f in self.fold_folders:
            m = PseLTae(**hyperparameters)

            if device == 'cpu':
                map_loc = 'cpu'
            else:
                map_loc = 'cuda:{}'.format(torch.cuda.current_device())
                m = m.cuda()
            d = torch.load(os.path.join(f, 'model.pth.tar'), map_location=map_loc)
            m.load_state_dict(d['state_dict'])
            self.model_instances.append(m)
        print('Successfully loaded {} model instances'.format(self.n_folds))

    def forward(self, input):
        """ Returns class logits
        Args:
            input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
                    Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
                    Pixel-Mask : Batch_size x Sequence length x Number of pixels
                    Extra-features : Batch_size x Sequence length x Number of features
        """
        with torch.no_grad():
            outputs = [F.log_softmax(m(input), dim=1) for m in self.model_instances]
            outputs = torch.stack(outputs, dim=0).mean(dim=0)
        return outputs

    def predict_class(self, input):
        """Returns class prediction
                Args:
            input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
                    Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
                    Pixel-Mask : Batch_size x Sequence length x Number of pixels
                    Extra-features : Batch_size x Sequence length x Number of features
        """
        with torch.no_grad():
            pred = self.forward(input).argmax(dim=1)
        return pred

class PseTae(nn.Module):
    """
    Pixel-Set encoder + Temporal Attention Encoder sequence classifier
    """

    def __init__(self, input_dim=10, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[132, 128], with_extra=True,
                 extra_size=4,
                 n_head=4, d_k=32, d_model=None, mlp3=[512, 128, 128], dropout=0.2, T=1000, len_max_seq=24,
                 positions=None,
                 mlp4=[128, 64, 32, 20], return_att=False):
        super(PseTae, self).__init__()
        self.spatial_encoder = PixelSetEncoder(input_dim, mlp1=mlp1, pooling=pooling, mlp2=mlp2, with_extra=with_extra,
                                               extra_size=extra_size)
        self.temporal_encoder = TemporalAttentionEncoder(in_channels=mlp2[-1], n_head=n_head, d_k=d_k,
                                                         d_model=d_model,
                                                         n_neurons=mlp3, dropout=dropout,
                                                         T=T, len_max_seq=len_max_seq, positions=positions,
                                                         return_att=return_att)
        self.decoder = get_decoder(mlp4)
        self.name = '_'.join([self.spatial_encoder.name, self.temporal_encoder.name])
        self.return_att = return_att

    def forward(self, input):
        """
         Args:
            input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
            Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
            Pixel-Mask : Batch_size x Sequence length x Number of pixels
            Extra-features : Batch_size x Sequence length x Number of features
        """
        out = self.spatial_encoder(input)
        if self.return_att:
            out, att = self.temporal_encoder(out)
            out = self.decoder(out)
            return out, att
        else:
            out = self.temporal_encoder(out)
            out = self.decoder(out)
            return out

    def param_ratio(self):
        total = get_ntrainparams(self)
        s = get_ntrainparams(self.spatial_encoder)
        t = get_ntrainparams(self.temporal_encoder)
        c = get_ntrainparams(self.decoder)

        print('TOTAL TRAINABLE PARAMETERS : {}'.format(total))
        print('RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%'.format(s / total * 100,
                                                                                          t / total * 100,
                                                                                          c / total * 100))
        return total



class PseGru(nn.Module):
    """
    Pixel-Set encoder + GRU
    """

    def __init__(self, input_dim=10, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[132, 128], with_extra=True,
                 extra_size=4, hidden_dim=128, mlp4=[128, 64, 32, 20], positions=None):
        super(PseGru, self).__init__()
        self.spatial_encoder = PixelSetEncoder(input_dim, mlp1=mlp1, pooling=pooling, mlp2=mlp2, with_extra=with_extra,
                                               extra_size=extra_size)
        self.temporal_encoder = GRU(in_channels=mlp2[-1], hidden_dim=hidden_dim, positions=positions)
        self.decoder = get_decoder(mlp4)
        self.name = '_'.join([self.spatial_encoder.name, self.temporal_encoder.name])

    def forward(self, input):
        """
         Args:
            input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
            Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
            Pixel-Mask : Batch_size x Sequence length x Number of pixels
            Extra-features : Batch_size x Sequence length x Number of features
        """
        out = self.spatial_encoder(input)

        out = self.temporal_encoder(out)
        out = self.decoder(out)
        return out

    def param_ratio(self):
        total = get_ntrainparams(self)
        s = get_ntrainparams(self.spatial_encoder)
        t = get_ntrainparams(self.temporal_encoder)
        c = get_ntrainparams(self.decoder)

        print('TOTAL TRAINABLE PARAMETERS : {}'.format(total))
        print('RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%'.format(s / total * 100,
                                                                                          t / total * 100,
                                                                                          c / total * 100))
        return total


class PseTempCNN(nn.Module):
    """
    Pixel-Set encoder + GRU
    """

    def __init__(self, input_dim=10, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[132, 128], with_extra=True,
                 extra_size=4, nker=[32, 32, 128], mlp3=[128, 128], seq_len=24, mlp4=[128, 64, 32, 20], positions=None):
        super(PseTempCNN, self).__init__()
        self.spatial_encoder = PixelSetEncoder(input_dim, mlp1=mlp1, pooling=pooling, mlp2=mlp2, with_extra=with_extra,
                                               extra_size=extra_size)
        self.temporal_encoder = TempConv(input_size=mlp2[-1], nker=nker, seq_len=seq_len, nfc=mlp3, positions=positions)
        self.decoder = get_decoder(mlp4)
        self.name = '_'.join([self.spatial_encoder.name, self.temporal_encoder.name])

    def forward(self, input):
        """
         Args:
            input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
            Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
            Pixel-Mask : Batch_size x Sequence length x Number of pixels
            Extra-features : Batch_size x Sequence length x Number of features
        """
        out = self.spatial_encoder(input)

        out = self.temporal_encoder(out)
        out = self.decoder(out)
        return out

    def param_ratio(self):
        total = get_ntrainparams(self)
        s = get_ntrainparams(self.spatial_encoder)
        t = get_ntrainparams(self.temporal_encoder)
        c = get_ntrainparams(self.decoder)

        print('TOTAL TRAINABLE PARAMETERS : {}'.format(total))
        print('RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%'.format(s / total * 100,
                                                                                          t / total * 100,
                                                                                          c / total * 100))
        return total


def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

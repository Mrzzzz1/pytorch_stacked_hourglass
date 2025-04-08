import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
class AttentionIter(nn.Module):
	"""docstring for AttentionIter"""
	def __init__(self, nChannels, LRNSize, IterSize):
		super(AttentionIter, self).__init__()
		self.nChannels = nChannels
		self.LRNSize = LRNSize
		self.IterSize = IterSize
		self.bn = nn.BatchNorm2d(self.nChannels)
		self.U = nn.Conv2d(self.nChannels, 1, 3, 1, 1)
		# self.spConv = nn.Conv2d(1, 1, self.LRNSize, 1, self.LRNSize//2)
		# self.spConvclone = nn.Conv2d(1, 1, self.LRNSize, 1, self.LRNSize//2)
		# self.spConvclone.load_state_dict(self.spConv.state_dict())
		_spConv_ = nn.Conv2d(1, 1, self.LRNSize, 1, self.LRNSize//2)
		_spConv = []
		for i in range(self.IterSize):
			_temp_ = nn.Conv2d(1, 1, self.LRNSize, 1, self.LRNSize//2)
			_temp_.load_state_dict(_spConv_.state_dict())
			_spConv.append(nn.BatchNorm2d(1))
			_spConv.append(_temp_)
		self.spConv = nn.ModuleList(_spConv)

	def forward(self, x):
		x = self.bn(x)
		u = self.U(x)
		out = u
		for i in range(self.IterSize):
			# if (i==1):
			# 	out = self.spConv(out)
			# else:
			# 	out = self.spConvclone(out)
			out = self.spConv[2*i](out)
			out = self.spConv[2*i+1](out)
			out = u + torch.sigmoid(out)
		return (x * out.expand_as(x))

class AttentionPartsCRF(nn.Module):
	"""docstring for AttentionPartsCRF"""
	def __init__(self, nChannels, LRNSize, IterSize, nJoints):
		super(AttentionPartsCRF, self).__init__()
		self.nChannels = nChannels
		self.LRNSize = LRNSize
		self.IterSize = IterSize
		self.nJoints = nJoints
		_S = []
		for _ in range(self.nJoints):
			_S_ = []
			_S_.append(AttentionIter(self.nChannels, self.LRNSize, self.IterSize))
			_S_.append(nn.BatchNorm2d(self.nChannels))
			_S_.append(nn.Conv2d(self.nChannels, 1, 1, 1, 0))
			_S.append(nn.Sequential(*_S_))
		self.S = nn.ModuleList(_S)

	def forward(self, x):
		out = []
		for i in range(self.nJoints):
			#out.append(self.S[i](self.attiter(x)))
		    out.append(self.S[i](x))
        return torch.cat(out, 1)
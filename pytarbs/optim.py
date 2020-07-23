import pytarbs.functions as f

class SGDMomentum:
	def __init__(self, lr=0.001, momentum=0.3):
		self.lr=lr
		self.momentum=momentum
		self.history=0

	def __call__(self, parameter, grad):
		self.history=self.history*momentum + (1-momentum)*grad
		return parameter - (self.history)*lr
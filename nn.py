import random
import pickle

"""
This module is providing a simple Kohonen Neural Network for
symbols recognision. Pay attention to standartize your symbols
images. You have to use the same size for all the symbol images.
Then you have to give pixels quantity for it to the network
when you create it (input_quant).
"""

class Input(object):
	"""
	A class for "input" neurons
	"""
	def __init__(self):
		self.outgoing_links = []
		

class Neuron(object):
	"""
	A class for neurons
	"""
	def __init__(self):
		self.incoming_links = []
		self.power = 0.0

class Link(object):
	"""
	A class for links between Inputs and Neurons
	"""
	def __init__(self, neuron, weight):
		self.neuron = neuron
		self.weight = weight

class KohonenNetwork(object):
	"""
	A class for all the Network
	"""
	def __init__(self, input_quant=1, symbols="0"):
		"""
		We have to give to our network the quantity of the inputs 
		(pixels in our case) and
		the list of symbols (for each of the symbol we create a neuron)
		"""
		self._inputs = []
		self._neurons = {}
		self.symbols = symbols
		for i in range(input_quant):
			self._inputs.append(Input())
		for sym in symbols:
			self._neurons[sym] = Neuron()
		for neur in self._neurons.values():
			for inp in self._inputs:
				link = Link(neur, random.randrange(1, stop=1000, step=1)/1000)
				neur.incoming_links.append(link)
				inp.outgoing_links.append(link)

	
	def handle(self, input_):
		"""
		Returns network`s answer to current image
		"""
		for i in range(len(self._inputs)):
			input_neuron = self._inputs[i]
			pixel = 1
			if input_[i] == 0:
				pixel = 2
			
			for outgoing_link in input_neuron.outgoing_links:
				outgoing_link.neuron.power += outgoing_link.weight * pixel
		maximum = '-'
		for neur in self._neurons.items():
			if neur[1].power > self._neurons[maximum].power:
				maximum = neur[0]
		for output_neuron in self._neurons.values():
			output_neuron.power = 0
		return maximum

	def study(self, input_, correct_answer):
		"""
		A basic method for weights corrections
		"""
		neuron = self._neurons[correct_answer]
		for i in range(len(neuron.incoming_links)):
			incoming_link = neuron.incoming_links[i]
			pixel = 1
			if input_[i] == 0:
				pixel = 2
			
			incoming_link.weight = incoming_link.weight + 0.5*(pixel - incoming_link.weight)

	def study_by_hands(self, input_, symbols):
		"""
		A method to learn your network by hands. Usually, I ran this method
		from the cycle where I was giving a new picture every time
		"""
		net_answer = self.handle(input_)
		print('Net things that this symbol is:', net_answer)
		print('To skip this, enter "skip"')
		print('If it`s wrong, input the right answer. If it`s right, press Enter:')
		my_answer = input()
		if my_answer == '':
			# self.study(input_, net_answer)
			return net_answer
		elif my_answer != net_answer and my_answer in symbols:
			self.study(input_, my_answer)
			return my_answer
		elif my_answer == 'skip':
			print('Skiped data')
			return ''
		else:
			print('You are not adectvative!')
			return ''

	def auto_study(self, input_, symbols, letter):
		"""
		A method to automatically learn your network. Usually, I ran this method
		from the cycle where I was giving a new picture every time
		"""
		net_answer = self.handle(input_)
		print('Net things that this symbol is:', net_answer)
		
		my_answer = letter
		print('Real is:', my_answer)

		if my_answer != net_answer and my_answer in symbols:
			self.study(input_, my_answer)
			
		else:
			print('You are not adectvative!')
			
		return net_answer

	def save_network(self):
		"""
		A method that saves the current network`s state to a "nn.pcl" file
		"""
		print('Saving network...')
		self.dumped = [self._inputs, self._neurons, self.symbols]
		with open('nn.pcl', 'wb') as f:
			pickle.dump(self.dumped, f)


def load_nn_from_file():
	"""
	A method that loads the current network`s state from a "nn.pcl" file.
	Before running this you have to check if the file exists
	"""
	print('Loading network...')
	with open('nn.pcl', 'rb') as f:
		dump = pickle.load(f)
		nn = KohonenNetwork()
		nn._inputs = dump[0]
		nn._neurons = dump[1]
		nn.symbols = dump[2]
		return nn

if __name__ == '__main__':
	symbols = ['-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 
	'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 
	'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
	network = KohonenNetwork(4380, symbols)
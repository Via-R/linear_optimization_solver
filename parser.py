import numpy as np
from fractions import Fraction

class InputParser:
	"""Reads file from input_path (parameter) and returns 
	its content via get_data()"""

	def __init__(self, input_path):
		with open(input_path) as f:
			#parsing first line goes here
			first_line = InputParser._format_to_math_form(f.readline())
			self._parse_first_line(first_line)

			last_line = ''
			for line in f:
				if line[0] == '\n':
					continue
				elif line[0] != '|':
					last_line = line
				
				#parsing main matrix goes here
			
			#parsing last line goes here
			self._parse_last_line(InputParser._format_to_math_form(last_line))
	@staticmethod
	def _format_to_math_form(line):
		"""Removes all spaces and adds omitted 'ones' in a given line"""
		return line.replace(' ', '').replace('-x', '-1x')

	def _parse_first_line(self, line):
		"""Gets a line and parses it into data concerning objective function
		Form of output: |numpy array of Fractions| [ { factor's fraction }, ... ]
		Index of each Fraction represents decremented index of corresponding variable
		Doesn't support error handling and constant terms"""

		raw_array = {} #basically it's a resulting array, but without order

		#Dividing the line using "+" as delimeter with further writing into the first line model named first_line_vect
		#Task_type model contains a string ("max" or "min"), depending on an input info
		line, self.task_type = line.split("=>")
		line = line[0] + line[1:].replace('-', '+-')
		op_arr = line.split('+')
		for i in op_arr:
			num, index = i[:-1].split("x[")
			raw_array[int(index)] = Fraction(num)

		self.first_line_vect = [Fraction(0,1)] * max(raw_array, key=int)
		for k, v in raw_array.items():
			self.first_line_vect[k - 1] = v
		self.first_line_vect = np.array(self.first_line_vect)

	def _parse_last_line(self, line):
		"""Gets a line and parses it into data concerning variables' conditions
		Form of output: |list of tuples| [ ( { index of inequality sign }, { condition's fraction } ), ... ]
		Index of each tuple represents decremented index of corresponding variable
		Expects variables not to have minus near them"""
		cond_list = line.split(",")
		op_list = ["<=", ">=", "<", ">", "arbitrary"]
		raw_dict = {}
		for expr in cond_list:
			op_index = 0
			for i in op_list:
				if i in expr:
					op_sym = i
					break
			f_tuple = op_list.index(op_sym), Fraction(expr[expr.find(op_sym)+len(op_sym):])
			raw_dict[int(expr[2:expr.find(op_sym)-1])] = f_tuple
		self.last_line_vect = [(4, 0)] * max(raw_dict, key=int)
		for k, v in raw_dict.items():
			self.last_line_vect[k - 1] = v	

	def get_data():
		"""Returns packed up info that was parsed using other methods"""
		pass

	def print_first_line(self):
		"""Prints out vector of objective function data"""
		print(self.first_line_vect)

	def print_last_line(self):
		"""Prints out list of variables' conditions"""
		print(self.last_line_vect)

if __name__ == "__main__":
	reader = InputParser('input')
	reader.print_first_line()

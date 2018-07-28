import numpy as np
from fractions import Fraction as Q

class InputParser:
	"""Reads file from input_path (parameter) and returns 
	its content via get_data()"""
	op_list = ["<=", ">=", "<", ">", "=", "arbitrary"]
	var_quantity = 0

	def __init__(self, input_path):
		with open(input_path) as f:
			#parsing first line goes here
			first_line = f.readline()
			while(first_line[0] == "#" or first_line[0] == "\n"):
				first_line = f.readline()
			first_line = InputParser._format_to_math_form(first_line)
			self.task_type, self.first_line_vect = self._parse_first_line(first_line)

			last_line = ''
			raw_matrix = []
			raw_constants = []
			self.inequalities = []
			for line in f:
				if line[0] == '\n' or line[0] == "#":
					continue
				elif line[0] != '|':
					last_line = line
					break
				
				#parsing main matrix goes here
				line = InputParser._format_to_math_form(line[1:])
				for i in InputParser.op_list:
					if i in line:
						self.inequalities.append(i)
						break
				curr_sym = self.inequalities[len(self.inequalities)-1]
				line = line[0] + line[1:line.find(curr_sym)].replace("-", "+-") + line[line.find(curr_sym):]

				parts_arr,  constant = line[:line.find(curr_sym)].split("+"), line[line.find(curr_sym)+len(curr_sym):]
				raw_constants.append(Q(constant))
				raw_dict = {}
				for i in parts_arr:
					num, ind = i[:-1].split("x[")
					raw_dict[int(ind)] = Q(num)
				raw_list = [0] * max(raw_dict, key=int)
				for k, v in raw_dict.items():
					raw_list[k - 1] = v
				raw_matrix.append(raw_list)

			for row in raw_matrix:
				if len(row) > InputParser.var_quantity:
					InputParser.var_quantity = len(row)
			for k, row in enumerate(raw_matrix):
				if len(row) < InputParser.var_quantity:
					for i in range(len(row), InputParser.var_quantity):
						raw_matrix[k].append(Q(0, 1))

			self.main_matrix = np.array(raw_matrix)
			self.constants_vector = np.array(raw_constants)

			#parsing last line goes here
			self.last_line_vect = self._parse_last_line(last_line.replace(' ',''))

	@staticmethod
	def _format_to_math_form(line):
		"""Removes all spaces and adds omitted 'ones' in a given line"""
		if line[0] == "x":
			line = "1" + line
		return line.replace(' ', '').replace('-x', '-1x').replace('+x', '+1x')

	def _parse_first_line(self, line):
		"""Gets a line and parses it into data concerning objective function
		Form of output: |numpy array of Qs| [ { factor's fraction }, ... ]
		Index of each Q represents decremented index of corresponding variable
		Doesn't support error handling and constant terms"""

		raw_array = {} #basically it's a resulting array, but without order

		#Dividing the line using "+" as delimeter with further writing into the first line model named first_line_vect
		#Task_type model contains a string ("max" or "min"), depending on an input info
		line, task_type = line[:line.find("=>")], line[line.find("=>")+2:-1]
		line = line[0] + line[1:].replace('-', '+-')
		op_arr = line.split('+')
		for i in op_arr:
			num, index = i[:-1].split("x[")
			raw_array[int(index)] = Q(num)

		first_line_vect = [Q(0,1)] * max(raw_array, key=int)
		for k, v in raw_array.items():
			first_line_vect[k - 1] = v
		return task_type, np.array(first_line_vect)

	def _parse_last_line(self, line):
		"""Gets a line and parses it into data concerning variables' conditions
		Form of output: |list of tuples| [ ( { index of inequality sign }, { condition's fraction } ), ... ]
		Index of each tuple represents decremented index of corresponding variable
		Expects variables not to have minus near them"""
		cond_list = line.split(",")
		
		raw_dict = {}
		for expr in cond_list:
			op_index = 0
			for i in InputParser.op_list:
				if i in expr:
					op_sym = i
					break
			f_tuple = op_sym, Q(expr[expr.find(op_sym)+len(op_sym):])
			raw_dict[int(expr[2:expr.find(op_sym)-1])] = f_tuple
		last_line_vect = [(4, 0)] * max(raw_dict, key=int)
		for k, v in raw_dict.items():
			last_line_vect[k - 1] = v
		return last_line_vect	

	def get_data(self):
		"""Returns a dictionary of the info that was parsed from the input file"""
		return {
			"objective_function": self.first_line_vect,
			"task_type": self.task_type,
			"last_conditions": self.last_line_vect,
			"matrix": self.main_matrix,
			"inequalities": self.inequalities,
			"constants": self.constants_vector,
		}

	def print_first_line(self):
		"""Prints out vector of objective function data"""
		print("First line: {}\n".format(self.first_line_vect))

	def print_task_type(self):
		print("Task type: {}\n".format(self.task_type))

	def print_last_line(self):
		"""Prints out list of variables' conditions"""
		print("Last line: {}\n".format(self.last_line_vect))

	def print_main_matrix(self):
		"""Prints out main matrix"""
		print("Matrix: {}\n".format(self.main_matrix))

	def print_constants(self):
		"""Prints out a vector of conditions' constants"""
		print("Constants' vector: {}\n".format(self.constants_vector))

	def print_inequalities(self):
		"""Prints out a list of inequality signs that concern the condition rows with corresponding decremented indexes"""
		print("Inequalities' vector: {}\n".format(self.inequalities))


# ------ Solver class section ------

class Solver:
	"""Основний клас, що містить спільні для всіх способів розв'язання методи та є базовим для класів,
	які відповідають різним способам розв'язання"""
	def __init__(self, input_data):
		if input_data['data_type'] == "file":
			reader_data = InputParser(input_data["data"]).get_data()
			self.objective_function = reader_data["objective_function"]
			self.task_type = reader_data["task_type"]
			self.last_conditions = reader_data["last_conditions"]
			self.matrix = reader_data["matrix"]
			self.inequalities = reader_data["inequalities"]
			self.constants = reader_data["constants"]
		else:
			print("This part has not been implemented yet")
		self.col_num = 0
		self.row_num = 2

	def make_basis_column(self):
		if self.matrix[self.row_num, self.col_num] == 0:
			print("Unexpected zero during basis making")
			return
		elif self.matrix[self.row_num, self.col_num] != 1:
			self.matrix[self.row_num] /= self.matrix[self.row_num, self.col_num]
		
		chosen_row = self.matrix[self.row_num]
		for i in [x for x in range(len(self.matrix)) if x != self.row_num]:
			self.matrix[i] -= chosen_row * self.matrix[i, self.col_num]


class SimplexSolver(Solver):
	"""Клас, що виконує розв'язання задачі лінійного програмування симплекс методом"""
	def __init__(self, input_data):
		super(SimplexSolver, self).__init__(input_data)
		self.make_basis_column()
		


# ------ Test section ------

import unittest
class TestParserMethods(unittest.TestCase):
	"""Tests for parsing class"""
	def test_math_formatting(self):
		"""Tests for valid expression formatting into a math form"""
		self.assertEqual(InputParser._format_to_math_form("- 9x[4] + 23x[1] -6x[2]+x[3] - x[5]=>max"), '-9x[4]+23x[1]-6x[2]+1x[3]-1x[5]=>max')

	def test_init(self):
		"""Tests for valid parsing of the input file"""
		dummy = InputParser('test_init')
		test_dict = {
			"objective_function": np.array([Q(-2, 1), Q(1, 1), Q(1, 1), Q(223, 1)]),
			"task_type": "max",
			"last_conditions": [(">", Q(0, 1)), (">=", Q(0, 1)), ("<", Q(3, 2)), ("<=", Q(2, 1))],
			"matrix": np.array([
				[Q(1, 1), Q(2, 1), Q(-3, 1), Q(-1, 1)],
				[Q(1, 1), Q(3, 1), Q(-2, 1), Q(0, 1)], 
				[Q(-4, 1), Q(-1, 1), Q(10, 1), Q(0, 1)], 
				[Q(1, 1), Q(-4, 1), Q(10, 1), Q(0, 1)]
			]),
			"inequalities": [">", "<=", "<", ">="],
			"constants": np.array([Q(4, 1), Q(0, 1), Q(-7, 1), Q(7, 2)])
		}
		for k, v in test_dict.items():
			self.assertTrue(np.array_equal(v, dummy.get_data()[k]))


if __name__ == "__main__":
	test_info = {"data_type": "file", "data": "test_init"}
	test = SimplexSolver(test_info)
	
	unittest.main()

import numpy as np
from fractions import Fraction as Q

def prmatr(m):
	"""Виводить матрицю у звичайному вигляді, без технічних символів та слів"""
	for i in m:
		for j in i:
			print(j, end=" ")
		print()

class InputParser:
	"""Клас для оброблення вхідної інформації з файлу або об'єкту
	Повертає оброблену інформацію через метод get_data()"""
	op_list = ["<=", ">=", "<", ">", "=", "arbitrary"]
	

	def __init__(self, input_path):
		with open(input_path) as f:
			#Обробка першого рядка з цільовою функцією
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
				
				#Обробка умов та заповнення відповідної їм матриці
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

			self.var_quantity = 0
			for row in raw_matrix:
				if len(row) > self.var_quantity:
					self.var_quantity = len(row)
			for k, row in enumerate(raw_matrix):
				if len(row) < self.var_quantity:
					for i in range(len(row), self.var_quantity):
						raw_matrix[k].append(Q(0, 1))

			self.main_matrix = np.array(raw_matrix)
			self.constants_vector = np.array(raw_constants)

			#Обробка останнього рядка з обмеженнями змінних
			self.last_line_vect = self._parse_last_line(last_line.replace(' ',''))

	@staticmethod
	def _format_to_math_form(line):
		"""Видаляє з рядка всі пробіли та додає одиничні множники де потрібно"""
		if line[0] == "x":
			line = "1" + line
		return line.replace(' ', '').replace('-x', '-1x').replace('+x', '+1x')

	def _parse_first_line(self, line):
		"""Отримує строку та обробляє її текст як інформацію про цільову функцію
		Форма виводу: |numpy array of Qs| [ { factor's fraction }, ... ]
		Індекс кожного Q відповідає декрементованому індексу відповідної змінної
		Не підтримує некоректну вхідну інформацію та константи в цільовій функції"""

		raw_array = {} #Результуючий масив, але невпорядкований

		#Розділення строки, використовуючи "+" як розділювач, з подальшим записом інформації в модель цільової функції в змінній first_line_vect
		#Змінна task_type містить строку ("max" або "min"), в залежності від вхідних даних
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
		"""Отримує строку та обробляє її як таку, що містить інформацію про загальні умови
		Форма виводу: |list of tuples| [ ( { index of inequality sign }, { condition's fraction } ), ... ]
		Індекс кожної пари відповідає декрементованому індексу відповідної змінної 
		Змінні не мають бути написані зі знаком "-" """
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
		"""Повертає об'єкт з усією обробленою інформацією, що була отримана з файлу"""
		return {
			"objective_function": self.first_line_vect,
			"task_type": self.task_type,
			"last_conditions": self.last_line_vect,
			"matrix": self.main_matrix,
			"inequalities": self.inequalities,
			"constants": self.constants_vector,
		}

	def print_first_line(self):
		"""Виводить вектор цільової функції"""
		print("First line: {}\n".format(self.first_line_vect))

	def print_task_type(self):
		"""Виводить тип задачі"""
		print("Task type: {}\n".format(self.task_type))

	def print_last_line(self):
		"""Виводить вектор обмежень змінних"""
		print("Last line: {}\n".format(self.last_line_vect))

	def print_main_matrix(self):
		"""Виводить основну матрицю"""
		print("Matrix: {}\n".format(self.main_matrix))

	def print_constants(self):
		"""Виводить вектор вільних змінних"""
		print("Constants' vector: {}\n".format(self.constants_vector))

	def print_inequalities(self):
		"""Виводить вектор знаків рівності\нерівності з системи початкових умов"""
		print("Inequalities' vector: {}\n".format(self.inequalities))


# ------ Solver class section ------

class Solver:
	"""Основний клас, що містить спільні для всіх способів розв'язання методи та є базовим для класів,
	які відповідають різним способам розв'язання"""
	def __init__(self, input_data):
		if input_data['data_type'] == "file":
			reader_data = InputParser(input_data["data"]).get_data()
			# print(reader_data)
			self.objective_function = reader_data["objective_function"]
			self.task_type = reader_data["task_type"]
			self.last_conditions = reader_data["last_conditions"]
			self.matrix = reader_data["matrix"]
			self.inequalities = reader_data["inequalities"]
			self.constants = reader_data["constants"]
		else:
			print("This part has not been implemented yet")
		self.col_num = 0
		self.row_num = 0
		self.basis = []
		self.basis_koef = np.array([])

		if self.task_type == "max":
			self.objective_function *= Q(-1)

	def _make_basis_column(self):
		"""Зводить задану в атрибутах колонку до одиничного вектора
		з одиницею на місці обраного в атрибутах рядка"""
		if self.matrix[self.row_num][self.col_num] == 0:
			print("Unexpected zero during basis making")
			return
		elif self.matrix[self.row_num][self.col_num] != 1:
			self.constants[self.row_num] /= self.matrix[self.row_num][self.col_num]
			self.matrix[self.row_num] /= self.matrix[self.row_num][self.col_num]
		
		chosen_row = self.matrix[self.row_num]
		for i in [x for x in range(len(self.matrix)) if x != self.row_num]:
			self.constants[i] -= self.constants[self.row_num] * self.matrix[i][self.col_num]
			self.matrix[i] -= chosen_row * self.matrix[i][self.col_num]

	def _make_conditions_equalities(self):
		"""Зводить всі нерівності умов до рівностей
		На даний момент не підтримуються строгі нерівності"""
		for i in range(len(self.inequalities)):
			if self.inequalities[i] == "<" or self.inequalities[i] == ">":
				print("This type of condition is not supported yet")
			elif self.inequalities[i] == ">=":
				self.matrix[i] *= Q(-1)
				self.constants[i] *= Q(-1)
				self.inequalities[i] = "<="
			if self.inequalities[i] == "<=":
				temp_matrix = []
				for j in range(len(self.matrix)):
					temp_matrix.append([Q(0)] * (len(self.matrix[0]) + 1))
				temp_matrix[i][-1] = Q(1)
				temp_matrix = np.array(temp_matrix)
				temp_matrix[:,:-1] = self.matrix
				self.matrix = temp_matrix
				self.inequalities[i] = "="

	def _get_basis_vectors_nums(self):
		"""Повертає список змінних, чиї вектори входять до одиничної підматриці матриці"""
		temp_matrix = self.matrix.T
		result = [-1] * len(temp_matrix[0])
		for i in range(len(temp_matrix)):
			num = -1
			for j in range(len(temp_matrix[i])):
				if temp_matrix[i][j] != 0 and temp_matrix[i][j] != 1:
					num = -1
					break
				if temp_matrix[i][j] == 1:
					if num == -1:
						num = j
					else:
						num = -1
						break
			if num > -1:
				result[num] = i
		return result

	def _set_basis_koef(self):
		"""Записує коефіцієнти базисних змінних в цільовій функції в окремий вектор"""
		self.basis[self.row_num] = self.col_num
		self.basis_koef[self.row_num] = self.objective_function[self.col_num]

	def _expand_objective_function_if_needed(self):
		"""Додає в цільову функцію штучні змінні з нульовим коефіцієнтом"""
		diff = len(self.matrix[0]) - len(self.objective_function)
		if diff > 0:
			num = len(self.objective_function)
			temp_array = [Q(0)] * (num + diff)
			temp_array[:num] = self.objective_function
			self.objective_function = np.array(temp_array)

class SimplexSolver(Solver):
	"""Виконує розв'язання задачі лінійного програмування симплекс методом"""
	def __init__(self, input_data):
		super(SimplexSolver, self).__init__(input_data)
		self.deltas = np.array([])
		self.thetas = np.array([])

	def print_all(self):
		"""Виводить всю доступну на даний момент інформацію про розвиток розв'язку задачі"""
		print(">------------------------------------------------------------<")
		print("Objective func: {}".format(self.objective_function))
		print("Basis constants: {}".format(self.basis_koef))
		print("Basis variables: {}".format(self.basis))
		print("Main matrix:\n-------------------------------")
		prmatr(self.matrix)
		print("-------------------------------\nConstants: {}".format(self.constants))
		print("Thetas: {}".format(self.thetas))
		print("Deltas: {}".format(self.deltas))
		print(">------------------------------------------------------------<\n")

	def _count_deltas(self):
		"""Розраховує вектор з дельтами"""
		temp_matrix = self.matrix.T
		temp_array = []
		for i in range(len(temp_matrix)):
			temp_array.append(self.objective_function[i] - temp_matrix[i].dot(self.basis_koef))
		self.deltas = np.array(temp_array)

	def _count_thetas(self):
		"""Розраховує вектор-стовпчик з відношеннями "тета" """
		self.thetas = self.constants / self.matrix.T[self.col_num]

	def _find_ind_of_min_theta(self):
		"""Знаходить індекс ведучого рядка, або повертає -1 якщо такого немає"""
		temp_min = 0
		min_set = False
		found_ind = -1
		for i in range(len(self.thetas)):
			if self.thetas[i] >= 0:
				temp_min = self.thetas[i]
				found_ind = i
				min_set = True
				break
		if min_set:
			for i in range(len(self.thetas)):
				if self.thetas[i] < 0:
					continue
				if self.thetas[i] < temp_min:
					temp_min = self.thetas[i]
					found_ind = i
		return found_ind

	def _make_constants_positive_if_needed(self):
		for i in self.constants:
			if i >= 0:
				return
		unset = True
		for i in range(len(self.constants)):
			for j in range(len(self.matrix[i])):
				if self.matrix[i][j] < 0:
					self.col_num = j
					self.row_num = i
					unset = False
					break
			if not unset:
				break
		if not unset:
			self._make_basis_column()
		self.basis = self._get_basis_vectors_nums()
		for i in range(len(self.basis)):
			self.basis_koef[i] = self.objective_function[self.basis[i]]


	def solve(self):
		"""Розв'язує задачу симплекс методом"""
		self.initial_variables_quantity = len(self.matrix[0])
		self._make_conditions_equalities()
		self.basis = self._get_basis_vectors_nums()
		for i in self.basis:
			if i == -1:
				print("Для подальших обчислень необхідна наявність одиничної підматриці")
				return
		self.basis_koef = np.array([0] * len(self.basis))
		self._expand_objective_function_if_needed()
		for i in range(len(self.basis)):
			self.basis_koef[i] = self.objective_function[self.basis[i]]
		self._make_constants_positive_if_needed()

		while True:
			self._count_deltas()
			min_delta = min(self.deltas)
			print(self.deltas)
			if min_delta < 0:
				self.col_num = int(np.where(self.deltas == min_delta)[0])
				self._count_thetas()
				self.row_num = self._find_ind_of_min_theta()
				if self.row_num == -1:
					print("Всі відношення \"тета\" від'ємні")
					return
				self._make_basis_column()
				self._set_basis_koef()
				self.print_all()
			else:
				break
		print("Done")
		final_result = [Q(0)] * len(self.matrix[0])
		for i in range(len(self.basis)):
			final_result[self.basis[i]] = self.constants[i]
		print("Final result (long): {}".format(final_result))
		print("Final result: {}".format(final_result[:self.initial_variables_quantity]))
		obj_func_val = self.objective_function.dot(np.array(final_result))
		if self.task_type == "max":
			obj_func_val *= -1
		print("Function value: {}".format(obj_func_val))

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
				[1, 2, -3, -1],
				[1, 3, -2, 0], 
				[-4, -1, 10, 0], 
				[1, -4, 10, 0]
			]),
			"inequalities": [">", "<=", "<", ">="],
			"constants": np.array([Q(4, 1), Q(0, 1), Q(-7, 1), Q(7, 2)])
		}
		for k, v in test_dict.items():
			self.assertTrue(np.array_equal(v, dummy.get_data()[k]))

class TestCommonLinearMethods(unittest.TestCase):
	"""Тести для класу Solver"""
	def __init__(self, *args, **kwargs):
		super(TestCommonLinearMethods, self).__init__(*args, **kwargs)
		self.input_info = {"data_type": "file", "data": "test_init"}
		self.input_info_main = {"data_type": "file", "data": "input"}

	def test_making_unit_basis(self):
		"""Тест на перевірку коректної роботи методу зведення стовпчика до одиничного вектора"""
		dummy = SimplexSolver(self.input_info)
		dummy._make_basis_column()
		test_matrix = np.array([
		 	[1, 2, -3, -1],
		 	[0, 1, 1, 1],
		 	[0, 7, -2, -4],
		 	[0, -6, 13, 1]
		 ])
		self.assertTrue(np.array_equal(test_matrix, dummy.matrix))

	def test_making_equalities_in_conditions(self):
		"""Тест на перевірку коректної роботи методу зведення нерівностей умов до рівностей"""
		dummy = SimplexSolver(self.input_info)
		for i in range(len(dummy.inequalities)):
			if len(dummy.inequalities[i]) == 1:
				dummy.inequalities[i] = ">=" if i % 2 == 0 else "<="
		before_test = dummy.matrix
		dummy._make_conditions_equalities()
		self.assertTrue(len(before_test[0]) + 4, len(dummy.matrix[0]))
		self.assertTrue(np.array_equal(np.array([
			[-1, -2, 3, 1, 1, 0, 0, 0],
			[1, 3, -2, 0, 0, 1, 0, 0],
			[4, 1, -10, 0, 0, 0, 1, 0],
			[-1, 4, -10, 0, 0, 0, 0, 1]
		]), dummy.matrix))

	def test_getting_basis_vectors_nums(self):
		"""Тест на перевірку коректної роботи методу отримання номерів змінних, що входять в базис"""
		dummy = SimplexSolver(self.input_info)
		correct_matrix = np.array([
			[2, 0, 0, 1],
			[2, 0, 1, 0],
			[2, 1, 0, 0]
		])
		incorrect_matrix = np.array([
			[3, 0, 0, 0],
			[3, 1, 1, 0],
			[3, 1, 2, 0],
			[3, 1, 0, 0]
		])
		dummy.matrix = correct_matrix
		self.assertTrue(np.array_equal(np.array([3, 2, 1]), dummy._get_basis_vectors_nums()))
		dummy.matrix = incorrect_matrix
		self.assertTrue(np.array_equal(np.array([-1, -1, -1, -1]), dummy._get_basis_vectors_nums()))


class TestSimplexMethod(unittest.TestCase):
	"""Тести для класу SimplexSolver"""
	def __init__(self, *args, **kwargs):
		super(TestSimplexMethod, self).__init__(*args, **kwargs)
		self.input_info = {"data_type": "file", "data": "test_init"}
		self.input_info_main = {"data_type": "file", "data": "input"}

	def test_for_right_solving(self):
		"""Тест на правильне розв'язання задачі симплекс методом"""
		dummy = SimplexSolver(self.input_info_main)
		dummy.solve()


if __name__ == "__main__":
	unittest.main()
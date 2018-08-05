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
	
	def __init__(self, input_info):
		inner_text = ""
		if input_info["data_type"] == "file":
			with open(input_info["data"]) as f:
				inner_text = f.read()
		elif input_info["data_type"] == "string":
			inner_text = input_info["data"]
		else:
			print("Unknown format of input data")

		inner_text = inner_text.replace('\t', '').replace(' ', '').split("\n")

		#Обробка першого рядка з цільовою функцією
		counter = 0
		first_line = inner_text[counter]
		while(first_line == '' or first_line[0] == '#'):
			counter += 1
			first_line = inner_text[counter]
		first_line = InputParser._format_to_math_form(first_line)
		self.task_type, self.first_line_vect = self._parse_first_line(first_line)

		last_cond = ''
		raw_matrix = []
		raw_constants = []
		self.inequalities = []
		for line in inner_text[counter + 1:]:
			if line == '' or line[0] == "#":
				continue
			elif line[:3] == ">>>":
				last_cond = ""
				break
			elif line[0] != '|':
				last_cond = line
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

		#Обробка рядка з обмеженнями змінних
		self.last_conditions = self._parse_last_cond(last_cond)
		#Обробка рядка з бажаним результатом розв'язку (використовується лише в тестуванні)
		self.result_list = []
		self.result = ""
		self.expected_error = ""
		counter = inner_text.index(last_cond) + 1
		last_line = ""
		if counter < len(inner_text):
			last_line = inner_text[counter]
		while(counter < len(inner_text) - 1 and last_line[:3] != '>>>'):
			counter += 1
			last_line = inner_text[counter]
		if counter >= len(inner_text) - 1 and last_line[:3] != '>>>':
			return
		raw_list, result, expected_error = self._parse_results(last_line)
		if raw_list != "":
			for i in raw_list.split(','):
				self.result_list.append(Q(i))
		self.result = result
		self.expected_error = expected_error


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
		line, task_type = line[:line.find("=>")], line[line.find("=>")+2:]
		line = line[0] + line[1:].replace('-', '+-')
		op_arr = line.split('+')
		for i in op_arr:
			num, index = i[:-1].split("x[")
			raw_array[int(index)] = Q(num)

		first_line_vect = [Q(0,1)] * max(raw_array, key=int)
		for k, v in raw_array.items():
			first_line_vect[k - 1] = v
		return task_type, np.array(first_line_vect)

	def _parse_last_cond(self, line):
		"""Отримує строку та обробляє її як таку, що містить інформацію про загальні умови
		Форма виводу: |list of tuples| [ ( { index of inequality sign }, { condition's fraction } ), ... ]
		Індекс кожної пари відповідає декрементованому індексу відповідної змінної 
		Змінні не мають бути написані зі знаком "-" """
		if line == "":
			return [["arbitrary", Q(0)]] * self.var_quantity	
		cond_list = line.split(",")
		raw_dict = {}
		for expr in cond_list:
			op_index = 0
			for i in InputParser.op_list:
				if i in expr:
					op_sym = i
					break
			f_pair = [op_sym, Q(expr[expr.find(op_sym)+len(op_sym):])]
			raw_dict[int(expr[2:expr.find(op_sym)-1])] = f_pair
		last_conditions = [[InputParser.op_list[5], Q(0)]] * max(raw_dict, key=int)
		for k, v in raw_dict.items():
			last_conditions[k - 1] = v
		complete_list = [["arbitrary", Q(0)]] * self.var_quantity
		complete_list[:len(last_conditions)] = last_conditions
		
		return complete_list

	def _parse_results(self, line):
		"""Отримує строку так обробляє її як таку, що містить інформацію про бажаний результат
		Інформація, отримана з цього методу використовується у тестуванні
		Форма виводу: |tuple| ( { масив значень відповідних змінних }, { значення цільової функції } )"""
		if not "(" in line:
			return "", "", line[3:]
		return line[line.find("(") + 1:line.find(")")], line[line.find("|") + 1:], ""

	def get_data(self):
		"""Повертає об'єкт з усією обробленою інформацією, що була отримана з файлу"""
		return {
			"objective_function": self.first_line_vect,
			"task_type": self.task_type,
			"last_conditions": self.last_conditions,
			"matrix": self.main_matrix,
			"inequalities": self.inequalities,
			"constants": self.constants_vector,
			"expected_vect": self.result_list,
			"expected_result": self.result,
			"error": self.expected_error
		}

	def print_first_line(self):
		"""Виводить вектор цільової функції"""
		print("First line: {}\n".format(self.first_line_vect))

	def print_task_type(self):
		"""Виводить тип задачі"""
		print("Task type: {}\n".format(self.task_type))

	def print_last_cond(self):
		"""Виводить вектор обмежень змінних"""
		print("Last line: {}\n".format(self.last_conditions))

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
		if input_data['data_type'] != "object":
			reader_data = InputParser(input_data).get_data()
			self.objective_function = reader_data["objective_function"]
			self.task_type = reader_data["task_type"]
			self.last_conditions = reader_data["last_conditions"]
			self.matrix = reader_data["matrix"]
			self.inequalities = reader_data["inequalities"]
			self.constants = reader_data["constants"]
			self.expected_vect = np.array(reader_data["expected_vect"])
			self.expected_result = Q(reader_data["expected_result"]) if reader_data["expected_result"] != "" else ""
			self.expected_error = reader_data["error"]
			self.result_error = ""
		else:
			print("This part has not been implemented yet")
		self.mute = input_data["mute"]
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
		self.artificial_variables = []

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
		"""Розраховує вектор-стовпчик з відношеннями "тета"
		Повертає False, якщо цільова функція необмежена на допустимій області"""
		#if np.count_nonzero(self.matrix.T[self.col_num]) == len(self.matrix.T[self.col_num]):
		#	self.thetas = self.constants / self.matrix.T[self.col_num]
		self.thetas = [Q(0)] * len(self.constants)
		for i in range(len(self.matrix)):
			if self.matrix[i][self.col_num] == 0:
				self.thetas[i] = -1
			elif self.matrix[i][self.col_num] < 0 and self.constants[i] == 0:
				self.thetas[i] = -1
			else:
				self.thetas[i] = self.constants[i] / self.matrix[i][self.col_num]
		# else:
			# return False
		# return True

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
		"""Якщо всі вільні члени від'ємні, то переходить до іншого базису"""
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

	def _add_artificial_basis(self):
		"""Створює одиничну підматрицю за допомогою штучних змінних М-методом"""
		M = np.amax(np.array(np.append(np.append(self.matrix, self.constants), self.objective_function))) + 1
		for i in range(len(self.basis)):
			if self.basis[i] == -1:
				temp_matrix = []
				for j in range(len(self.matrix)):
					temp_matrix.append([Q(0)] * (len(self.matrix[0]) + 1))
				temp_matrix[i][-1] = Q(1)
				temp_matrix = np.array(temp_matrix)
				temp_matrix[:,:-1] = self.matrix
				self.matrix = temp_matrix
				self.objective_function = np.append(self.objective_function, M)
				self.artificial_variables.append(len(self.objective_function) - 1)

	def _normalize_conditions(self):
		"""Зводить задачу до аналогічної, у якій всі змінні більше або рівні нулю"""
		self.substitution_queue = []
		self.arbitrary_pairs = []

		for i in range(len(self.last_conditions)):
			if len(self.last_conditions[i][0]) == 1:
				return False

			elif self.last_conditions[i][0] == "<=":
				for j in range(len(self.matrix)):
					self.matrix[j][i] *= -1
				self.substitution_queue.append((i, "*=-1"))
				self.objective_function[i] *= -1
				self.last_conditions[i] = [">=", self.last_conditions[i][1] * -1]
			if self.last_conditions[i][0] == ">=":
				if self.last_conditions[i][1] != 0:
					for j in range(len(self.matrix)):
						self.constants[j] -= self.matrix[j][i] * self.last_conditions[i][1]
					self.substitution_queue.insert(0, (i, "+={}".format(self.last_conditions[i][1])))
					self.last_conditions[i][1] = Q(0)

			if self.last_conditions[i][0] == "arbitrary":
				new_pair = i, len(self.matrix[0])
				new_matrix = []
				for j in range(len(self.matrix)):
					new_matrix.append([Q(0)] * (len(self.matrix[0]) + 1))
				for j in range(len(self.matrix)):
					new_matrix[j][-1] = -self.matrix[j][i]

				
				new_matrix = np.array(new_matrix)
				new_matrix[:,:-1] = self.matrix
				self.matrix = new_matrix
				
				self.objective_function = np.append(self.objective_function, -self.objective_function[i])
				self.last_conditions[i] = [">=", Q(0)]
				self.arbitrary_pairs.append(new_pair)

		return True

	def _cancel_subtitution(self):
		"""Повертає початкові значення змінним, якщо відбулася заміна"""
		for i in self.substitution_queue:
			exec("self.final_result[i[0]]" + i[1])
			if "*" in i[1]:
				self.objective_function[i[0]] *= Q(i[1][2:])

		for i in self.arbitrary_pairs:
			self.final_result[i[0]] -= self.final_result[i[1]]


	# def _make_matrix_rectangular(self):
	# 	"""Після додавання нових змінних до основних умов, додає їх в усі 
	# 	рядки нерівностей, але з нульовими множниками"""
	# 	max_len = 0
	# 	for i in self.matrix:
	# 		if len(i) > max_len:
	# 			max_len = len(i)
	# 	for i in range(len(self.matrix)):
	# 		if len(self.matrix[i]) < max_len:
	# 			to_insert = [Q(0)] * (max_len - len(self.matrix[i]))
	# 			self.matrix[i] = np.append(self.matrix[i], to_insert)

	def _get_col_num(self, indices_list):
		"""Повертає індекс ведучого стовпчика, засновуючись на векторі з дельтами"""
		if len(indices_list) == 1:
			return indices_list[0]
		for i in range(len(indices_list)):
			temp_thetas = []
			for j in range(len(self.matrix)):
				if self.matrix[j][indices_list[i]] == 0 or (self.constants[j] == 0 and self.matrix[j][indices_list[i]] < 0):
					temp_thetas.append(-1)
				else:
					temp_thetas.append(self.constants[j] / self.matrix[j][indices_list[i]])
			for j in temp_thetas:
				if j >= 0:
					break
			else:
				indices_list[i] = -1
		for i in indices_list:
			if i >= 0:
				return i
		return -1

	def _check_for_ambiguous_result(self):
		"""Перевіряє чи відповідає небазисній змінній нульова дельта (якщо штучна змінна базисна, її пара теж вважається базисною)"""
		basis = set(self.basis)
		for i in self.arbitrary_pairs:
			if i[0] in basis:
				basis.add(i[1])
			elif i[1] in basis:
				basis.add(i[0])
		non_basis_set = set(range(len(self.objective_function))) - basis
		for i in non_basis_set:
			if self.deltas[i] == 0:
				self.result_error = "infinite"
				raise SolvingError("Базисній змінній відповідає нульова дельта:\nІснує нескінченна кількість розв'язків")

	def _check_for_empty_allowable_area(self):
		"""Перевіряє чи є у кінцевому векторі з множниками змінних штучна змінна з відмнінним від нуля множником"""
		for i in self.artificial_variables:
			if self.final_result[i] != 0:
				self.result_error = "empty"
				raise SolvingError("В оптимальному розв'язку присутня штучна змінна:\nДопустима область порожня")

	def solve(self):
		"""Розв'язує задачу симплекс методом"""
		self.initial_variables_quantity = len(self.matrix[0])
		if not self._normalize_conditions():
			raise SolvingError("В заданих умовах обмеження змінних містять строгі знаки нерівностей або знак рівності - дані вхідні дані некоректні для симплекс методу")
		self._make_conditions_equalities()
		self.basis = self._get_basis_vectors_nums()
		for i in self.basis:
			if i == -1:
				self._add_artificial_basis()
				break
		self.basis_koef = np.array([0] * len(self.basis))
		self._expand_objective_function_if_needed()
		for i in range(len(self.basis)):
			self.basis_koef[i] = self.objective_function[self.basis[i]]
		self._make_constants_positive_if_needed()

		safety_counter = 0
		while True:
			safety_counter += 1
			if safety_counter > 100:
				raise SolvingError("Кількість ітерацій завелика, цикл зупинено")

			self._count_deltas()
			min_delta = min(self.deltas)
			if min_delta < 0:
				self.col_num = self._get_col_num(np.where(self.deltas == min_delta)[0].tolist())
				if self.col_num == -1:
					self.result_error = "unlimited"
					raise SolvingError("Неможливо обрати ведучий стовпчик, всі стовпчики з від'ємними дельта утворюють від'ємні тета:\nЦільова функція необмежена на допустимій області")
				self._count_thetas()
				self.row_num = self._find_ind_of_min_theta()
				if self.row_num == -1:
					self.result_error = "unlimited"
					raise SolvingError("Всі тета від'ємні:\nЦільова функція необмежена на допустимій області")
				self._make_basis_column()
				self._set_basis_koef()
				if not self.mute: self.print_all()
			else:
				for i in range(len(self.constants)):
					if self.constants[i] < 0:
						another_iteration = False
						for j in range(len(self.matrix[i])):
							if self.matrix[i][j] < 0:
								self.row_num = i
								self.col_num = j
								self._set_basis_koef()
								self._make_basis_column()
								another_iteration  = True
								break
						if another_iteration:
							if not self.mute: self.print_all()
							break
						else:
							self.result_error = "empty"
							raise SolvingError("Неможливо отримати опорний розв'язок, всі дельта не менші нуля, але не всі значення базисних змінних більші рівні нулю:\nДопустима область порожня")
				else:
					break
		self.final_result = [Q(0)] * len(self.matrix[0])
		for i in range(len(self.basis)):
			self.final_result[self.basis[i]] = self.constants[i]

		if self.task_type == "max":
			self.objective_function *= -1

		self._cancel_subtitution()

		self.result_vect = self.final_result[:self.initial_variables_quantity]
		obj_func_val = self.objective_function[:self.initial_variables_quantity].dot(np.array(self.result_vect))
		self.result = obj_func_val
		self._check_for_ambiguous_result()
		self._check_for_empty_allowable_area()
		if not self.mute: 
			print("Done")
			print("Final result (long): {}".format(self.final_result))
			print("Final result: {}".format(self.result_vect))
			print("Function value: {}".format(obj_func_val))


# ------ Custom exception section ------


class SolvingError(Exception):
	def __init__(self, message):
		super().__init__(message)


# ------ Test section ------


test_input_string = """
# Not suitable for calculations 

x[2]+x[3]-2x[1]+223x[4] =>max

|2x[2]+x[1]-3x[3]-x[4]>4
|-2x[3]+x[1]+3x[2]<=0
|-x[2]+10x[3]-4x[1]<-7
|x[1]+10x[3]-4x[2]>=7/2

x[1]>0, x[2]>=0, x[3]<3/2, x[4]<=2
"""

import unittest
class TestParserMethods(unittest.TestCase):
	"""Tests for parsing class"""
	def test_math_formatting(self):
		"""Tests for valid expression formatting into a math form"""
		self.assertEqual(InputParser._format_to_math_form("- 9x[4] + 23x[1] -6x[2]+x[3] - x[5]=>max"), '-9x[4]+23x[1]-6x[2]+1x[3]-1x[5]=>max')

	def test_input(self):
		"""Tests for valid parsing of the input file"""
		dummy = InputParser({'data_type': 'string','data':test_input_string})
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
		self.input_info = {'data_type': 'string','data':test_input_string, "mute": True}
		self.input_info_main = {"data_type": "file", "data": "input", "mute": True}

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
		self.input_info = {'data_type': 'string','data':test_input_string, "mute": True}
		self.input_info_main = {"data_type": "file", "data": "input", "mute": True}

	def test_for_choosing_column(self):
		"""Тест для методу, що обирає ведучий стовпчик (_get_col_num)"""
		dummy = SimplexSolver(self.input_info_main)
		correct_matrix = np.array([
			[1, -1, -2],
			[0, -1, -2],
			[0, -1, 2]
		])
		incorrect_matrix = np.array([
			[1, -1, -2],
			[0, -1, -2],
			[0, -1, -2]
		])
		dummy.constants = [0, 1, 1]
		dummy.matrix = correct_matrix
		self.assertEqual(2, dummy._get_col_num([1, 2]))
		dummy.matrix = incorrect_matrix
		self.assertEqual(-1, dummy._get_col_num([1, 2]))

	def test_for_right_solving(self):
		"""Тест на правильне розв'язання різних задач симплекс методом"""
		with open("test_input") as f:
			inner_text = f.read()
			inner_text = inner_text.split("***")
			for i in inner_text:
				dummy = SimplexSolver({"data_type":"string", "data": i, "mute":True})
				try:
					dummy.solve()
				except SolvingError as err:
					self.assertEqual(dummy.expected_error, dummy.result_error)
				else:
					self.assertEqual(dummy.expected_result, dummy.result)
					# if dummy.result_error == "":
					# else:
						

if __name__ == "__main__":
	unittest.main()
	data_to_solve = """
	x[1]+x[2]=>max
	|x[1]+x[2]<=1
	|x[1]+x[2]>=2
	"""

	dummy = SimplexSolver({"data_type":"string", "data": data_to_solve, "mute":False})
	try:
		dummy.solve()
	except SolvingError as err:
		print(err)
	else:
		print("OK")
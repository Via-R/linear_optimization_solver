class InputParser:
	"""Reads file from input_path (parameter) and returns 
	its content via get_text()"""
	def __init__(self, input_path):
		with open(input_path) as f:
			first_line = f.readline()
			#parsing first line goes here

			last_line = ''
			for line in f:
				if line[0] == '\n':
					continue
				elif line[0] != '|':
					last_line = line
				
				#parsing main matrix goes here
			
			#parsing last line goes here
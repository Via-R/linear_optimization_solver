from linearsolver import SimplexSolver

solver = SimplexSolver("file", "input")
result = solver.get_result()
with open("output.html", "w") as f: 
	f.write(result) 
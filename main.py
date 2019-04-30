from linearsolver import DualSimplexSolver

solver = DualSimplexSolver("file", "input")
result = '<link rel="stylesheet" type="text/css" href="local.css"/>' + solver.get_result()
with open("output.html", "w") as f: 
	f.write(result) 
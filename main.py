from linearsolver import SimplexSolver

input_info = {"data_type": "file", "data": "input"}
solver = SimplexSolver(input_info)
solver.solve()
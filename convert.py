import json

files = ["AI.ipynb"]

for file in files:
    code = json.load(open(file))
    py_file = open(f"{file}.py", "w+")

    for cell in code["cells"]:
        if cell["cell_type"] == "code":
            for line in cell["source"]:
                py_file.write(line)
            py_file.write("\n")

    py_file.close()

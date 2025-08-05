import sys, os

path = os.path.abspath(os.path.dirname(__file__))
print(path)

filepath = path + os.path.sep + 'prompt_1.txt'
mode = 'r'

result = ''
with open(filepath, mode) as f:
    result = f.read()

print(result)


import pathlib
import sys
x=pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0,x)
print(x)
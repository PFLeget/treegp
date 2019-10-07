import os
import glob

tests = glob.glob("test_*")
for test in tests:
    os.system("python %s"%(test))


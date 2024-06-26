import Dcluster as dcl
import matplotlib

matplotlib.use("TkAgg")

dcl.run(fi="data_array.dat", sep=" ", percent=5)

import Dcluster as dcl
import matplotlib

matplotlib.use("TkAgg")

dcl.run(fi="lcss_distance/data_array.dat", sep=" ", percent=5)

from scipy.io import loadmat, savemat

ds = {} #digitStruct.mat
for i in range(33402):
    mat = loadmat(f"mat/{i+1}.png_bbox.mat")
    ds[f"{i+1}.png"] = mat["bbox"]

savemat("digitStruct_(1).mat", ds)
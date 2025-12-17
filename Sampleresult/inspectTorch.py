import torch

FILE = r"C:\Users\SABRA\Documents\GitHub\Visual-Place-Recognition-Project\result sample\000.torch"

data = torch.load(FILE, map_location="cpu", weights_only=False)

print("\nInlier counts per retrieval rank:\n")

for i, item in enumerate(data):
    num_inliers = item.get("num_inliers", 0)
    print(f"Top-{i+1:02d} inliers count : {num_inliers}")

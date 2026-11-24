from datasets import load_dataset
ds = load_dataset("flwrlabs/femnist", split="train")
print(ds)
print(ds[0].keys())
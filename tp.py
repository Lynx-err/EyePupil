from collections import Counter
ds = EyeDentifyPPDataset(data_root, sr_method, split='train', transform=transform)
counts = Counter([s['eye_side'] for s in ds.samples])
print(counts)  # Expect something like: {'left_eyes': N1, 'right_eyes': N2}
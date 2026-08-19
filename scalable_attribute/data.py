import os

from data_utils.dataloaders.attribute_dataloader import PCDataset


def h5_files(root, manifest=None):
    if manifest:
        manifest = os.path.expanduser(os.path.expandvars(manifest))
        with open(manifest, encoding="utf-8") as handle:
            entries = [line.strip() for line in handle if line.strip()]
        files = [path if os.path.isabs(path) else os.path.join(root, path)
                 for path in entries]
        missing = [path for path in files if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError("Manifest HDF5 not found: " + missing[0])
    else:
        files = []
        for directory, _, names in os.walk(root):
            files.extend(os.path.join(directory, name)
                         for name in names if name.endswith(".h5"))
        files.sort()
    if not files:
        raise RuntimeError("No HDF5 files selected under " + root)
    if any(not path.endswith(".h5") for path in files):
        raise ValueError("HDF5 manifest contains a non-.h5 entry")
    return files


class UncachedPCDataset(PCDataset):
    """Use the public V1 sample semantics without retaining a full epoch in RAM."""

    def __getitem__(self, index):
        item = super().__getitem__(index)
        self.cache.clear()
        return item

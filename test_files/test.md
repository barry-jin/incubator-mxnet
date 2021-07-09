```{.python .input}
# Pre-processing transformers
training_transformer = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                 saturation=jitter_param),
    transforms.RandomLighting(lighting_param),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

validation_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# loading the data and apply pre-processing(transforms) on images
train_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(train_path).transform_first(training_transformer),
    batch_size=batch_size, shuffle=True, num_workers=num_workers, try_nopython=True)

val_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(val_path).transform(validation_transformer),
    batch_size=batch_size, shuffle=False, num_workers=num_workers, try_nopython=False)

test_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(test_path).transform_first(validation_transformer),
    batch_size=batch_size, shuffle=False, num_workers=num_workers, try_nopython=None)
```
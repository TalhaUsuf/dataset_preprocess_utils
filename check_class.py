from dataset_cls import identities_ds
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 

print(dir(A))
trf = A.Compose(
    [
        A.Resize(always_apply=True, p=1.0, height=224, width=224, interpolation=3),
        A.ElasticTransform(p=0.5, alpha=2.679999828338623, sigma=21.479999542236328, alpha_affine=10.739999771118164, interpolation=1, border_mode=1, value=(0, 0, 0), mask_value=None, approximate=False),
        A.Equalize(mode='cv', by_channels=True, p=0.5),
        A.CLAHE(clip_limit=(1, 6), tile_grid_size=(8,8), p=0.5),
        A.OpticalDistortion(always_apply=False, p=0.7, distort_limit=(-0.30000001192092896, 0.8299999833106995), shift_limit=(-0.05000000074505806, 0.05000000074505806), interpolation=3, border_mode=1, value=(0, 0, 0), mask_value=None),
        A.HorizontalFlip(always_apply=False, p=0.7),
        ToTensorV2()
    ]
)
ds = identities_ds(csv="valid.csv", transform=trf)

for k in ds:
    print(k[1])
    break
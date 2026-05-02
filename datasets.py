import os
import random
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset, ConcatDataset, Dataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset
from domainbed.lib.openset_manifest import load_samples, load_uid_split, index_samples_by_uid, select_samples

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = ["Debug28", "Debug224", "ColoredMNIST", "RotatedMNIST", "VLCS", "PACS", "OfficeHome", "TerraIncognita", "DomainNet", "SVIRO", "WILDSCamelyon", "WILDSFMoW", "SpawriousO2O_easy", "SpawriousO2O_medium", "SpawriousO2O_hard", "SpawriousM2M_easy", "SpawriousM2M_medium", "SpawriousM2M_hard", "OpenSetDomainNetObjects",]

def get_dataset_class(dataset_name):
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)

class MultipleDomainDataset:
    N_STEPS = 5001
    CHECKPOINT_FREQ = 100
    N_WORKERS = 8
    ENVIRONMENTS = None
    INPUT_SHAPE = None

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(TensorDataset(torch.randn(16, *self.INPUT_SHAPE), torch.randint(0, self.num_classes, (16,))))

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']

class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape, num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')
        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)
        original_images = torch.cat((original_dataset_tr.data, original_dataset_te.data))
        original_labels = torch.cat((original_dataset_tr.targets, original_dataset_te.targets))
        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]
        self.datasets = []
        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))
        self.input_shape = input_shape
        self.num_classes = num_classes

class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9], self.color_dataset, (2, 28, 28,), 2)
        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        labels = (labels < 5).float()
        labels = self.torch_xor_(labels, self.torch_bernoulli_(0.25, len(labels)))
        colors = self.torch_xor_(labels, self.torch_bernoulli_(environment, len(labels)))
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0
        x = images.float().div_(255.0)
        y = labels.view(-1).long()
        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75], self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([transforms.ToPILImage(), transforms.Lambda(lambda x: rotate(x, angle, fill=(0,), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)), transforms.ToTensor()])
        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])
        y = labels.view(-1)
        return TensorDataset(x, y)

class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)
        transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        augment_transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.3, 0.3, 0.3, 0.3), transforms.RandomGrayscale(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        self.datasets = []
        for i, environment in enumerate(environments):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform
            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path, transform=env_transform)
            self.datasets.append(env_dataset)
        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class WILDSEnvironment:
    def __init__(self, wilds_dataset, metadata_name, metadata_value, transform=None):
        self.name = metadata_name + "_" + str(metadata_value)
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(metadata_array[:, metadata_index] == metadata_value)[0]
        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)
        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)

class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)

    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        augment_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.3, 0.3, 0.3, 0.3), transforms.RandomGrayscale(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        self.datasets = []
        for i, metadata_value in enumerate(self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform
            env_dataset = WILDSEnvironment(dataset, metadata_name, metadata_value, env_transform)
            self.datasets.append(env_dataset)
        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))

class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3", "hospital_4"]

    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)

class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3", "region_4", "region_5"]

    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(dataset, "region", test_envs, hparams['data_augmentation'], hparams)



class SimpleImageListDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(sample["label"], dtype=torch.long), sample["uid"]

    def get_metadata(self, index):
        return self.samples[index]


class ManifestLabeledDataset(SimpleImageListDataset):
    pass


class ManifestEvalDataset(SimpleImageListDataset):
    pass


class ManifestUnlabeledDataset(Dataset):
    def __init__(self, samples, weak_transform=None, strong_transform=None, mask_transform=None):
        self.samples = samples
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample["path"]).convert("RGB")
        x_weak = self.weak_transform(image) if self.weak_transform else image
        x_strong = self.strong_transform(image) if self.strong_transform else x_weak
        x_mask = self.mask_transform(image) if self.mask_transform else x_weak
        y_true = torch.tensor(sample["label"], dtype=torch.long)
        return x_weak, x_strong, x_mask, y_true, sample["uid"]


def collect_imagefolder_samples(root, keep_classes=None, label_map=None):
    dataset = ImageFolder(root=root)
    samples = []
    for path, old_y in dataset.samples:
        class_name = dataset.classes[old_y]
        if keep_classes is not None and class_name not in keep_classes:
            continue
        if label_map is None:
            new_y = old_y
        else:
            new_y = label_map[class_name]
        samples.append((path, new_y))
    return samples




def wrap_samples(raw_samples, env_name, source_name, is_ood, uid_prefix):
    wrapped = []
    for i, (path, label) in enumerate(raw_samples):
        wrapped.append({
            "path": path,
            "label": label,
            "is_ood": is_ood,
            "source": source_name,
            "env": env_name,
            "uid": f"{uid_prefix}:{i}"
        })
    return wrapped
def sample_n(samples, n, seed=0):
    rng = random.Random(seed)
    samples = list(samples)
    rng.shuffle(samples)
    return samples[:n]

class CustomImageFolder(Dataset):
    def __init__(self, folder_path, class_index, limit=None, transform=None):
        self.folder_path = folder_path
        self.class_index = class_index
        self.image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        if limit:
            self.image_paths = self.image_paths[:limit]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.class_index, dtype=torch.long)
        return img, label

class SpawriousBenchmark(MultipleDomainDataset):
    ENVIRONMENTS = ["Test", "SC_group_1", "SC_group_2"]
    input_shape = (3, 224, 224)
    num_classes = 4
    class_list = ["bulldog", "corgi", "dachshund", "labrador"]

    def __init__(self, train_combinations, test_combinations, root_dir, augment=True, type1=False):
        self.type1 = type1
        root_dir = os.path.join(root_dir, "spawrious224")
        train_datasets, test_datasets = self._prepare_data_lists(train_combinations, test_combinations, root_dir, augment)
        self.datasets = [ConcatDataset(test_datasets)] + train_datasets

    def _prepare_data_lists(self, train_combinations, test_combinations, root_dir, augment):
        test_transforms = transforms.Compose([transforms.Resize((self.input_shape[1], self.input_shape[2])), transforms.transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        if augment:
            train_transforms = transforms.Compose([transforms.Resize((self.input_shape[1], self.input_shape[2])), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.3, 0.3, 0.3, 0.3), transforms.RandomGrayscale(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        else:
            train_transforms = test_transforms
        train_data_list = self._create_data_list(train_combinations, root_dir, train_transforms)
        test_data_list = self._create_data_list(test_combinations, root_dir, test_transforms)
        return train_data_list, test_data_list

    def _create_data_list(self, combinations, root_dir, transforms):
        data_list = []
        if isinstance(combinations, dict):
            for_each_class_group = []
            cg_index = 0
            for classes, comb_list in combinations.items():
                for_each_class_group.append([])
                for ind, location_limit in enumerate(comb_list):
                    if isinstance(location_limit, tuple):
                        location, limit = location_limit
                    else:
                        location, limit = location_limit, None
                    cg_data_list = []
                    for cls in classes:
                        path = os.path.join(root_dir, f"{0 if not self.type1 else ind}/{location}/{cls}")
                        data = CustomImageFolder(folder_path=path, class_index=self.class_list.index(cls), limit=limit, transform=transforms)
                        cg_data_list.append(data)
                    for_each_class_group[cg_index].append(ConcatDataset(cg_data_list))
                cg_index += 1
            for group in range(len(for_each_class_group[0])):
                data_list.append(ConcatDataset([for_each_class_group[k][group] for k in range(len(for_each_class_group))]))
        else:
            for location in combinations:
                path = os.path.join(root_dir, f"{0}/{location}/")
                data = ImageFolder(root=path, transform=transforms)
                data_list.append(data)
        return data_list

    def build_type1_combination(self,group,test,filler):
        total = 3168
        counts = [int(0.97*total),int(0.87*total)]
        combinations = {}
        combinations['train_combinations'] = {("bulldog",):[(group[0],counts[0]),(group[0],counts[1])], ("dachshund",):[(group[1],counts[0]),(group[1],counts[1])], ("labrador",):[(group[2],counts[0]),(group[2],counts[1])], ("corgi",):[(group[3],counts[0]),(group[3],counts[1])], ("bulldog","dachshund","labrador","corgi"):[(filler,total-counts[0]),(filler,total-counts[1])],}
        combinations['test_combinations'] = {("bulldog",):[test[0], test[0]], ("dachshund",):[test[1], test[1]], ("labrador",):[test[2], test[2]], ("corgi",):[test[3], test[3]],}
        return combinations

    def build_type2_combination(self,group,test):
        total = 3168
        counts = [total,total]
        combinations = {}
        combinations['train_combinations'] = {("bulldog",):[(group[0],counts[0]),(group[1],counts[1])], ("dachshund",):[(group[1],counts[0]),(group[0],counts[1])], ("labrador",):[(group[2],counts[0]),(group[3],counts[1])], ("corgi",):[(group[3],counts[0]),(group[2],counts[1])],}
        combinations['test_combinations'] = {("bulldog",):[test[0], test[1]], ("dachshund",):[test[1], test[0]], ("labrador",):[test[2], test[3]], ("corgi",):[test[3], test[2]],}
        return combinations

class SpawriousO2O_easy(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ["desert","jungle","dirt","snow"]
        test = ["dirt","snow","desert","jungle"]
        filler = "beach"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'], type1=True)

class SpawriousO2O_medium(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['mountain', 'beach', 'dirt', 'jungle']
        test = ['jungle', 'dirt', 'beach', 'snow']
        filler = "desert"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'], type1=True)

class SpawriousO2O_hard(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['jungle', 'mountain', 'snow', 'desert']
        test = ['mountain', 'snow', 'desert', 'jungle']
        filler = "beach"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'], type1=True)

class SpawriousM2M_easy(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['desert', 'mountain', 'dirt', 'jungle']
        test = ['dirt', 'jungle', 'mountain', 'desert']
        combinations = self.build_type2_combination(group,test)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation']) 

class SpawriousM2M_medium(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['beach', 'snow', 'mountain', 'desert']
        test = ['desert', 'mountain', 'beach', 'snow']
        combinations = self.build_type2_combination(group,test)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'])
        
class SpawriousM2M_hard(SpawriousBenchmark):
    ENVIRONMENTS = ["Test","SC_group_1","SC_group_2"]
    def __init__(self, root_dir, test_envs, hparams):
        group = ["dirt","jungle","snow","beach"]
        test = ["snow","beach","dirt","jungle"]
        combinations = self.build_type2_combination(group,test)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'])


class OpenSetDomainNetObjects(MultipleDomainDataset):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["A_real", "A_painting", "B_mix", "B_id_eval", "B_ood_eval", "T_clipart"]
    UNLABELED_ENVS = [2]
    EVAL_ONLY_ENVS = [3, 4, 5]
    OOD_EVAL_ENVS = [4]
    INPUT_SHAPE = (3, 224, 224)

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        manifest_root = os.path.join(root, "openset_domainnet_objects_v1")
        if not os.path.isdir(manifest_root):
            raise FileNotFoundError(f"Missing benchmark manifest root: {manifest_root}. Build it with scripts/build_openset_domainnet.py")
        self.id_classes = ["book", "clock", "keyboard", "lamp", "mug", "scissors"]
        self.num_classes = len(self.id_classes)
        eval_t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        label_t = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.3, 0.3, 0.3, 0.3), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        weak_t = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        strong_t = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.4, 0.4, 0.4, 0.2), transforms.RandomGrayscale(p=0.2), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        mask_t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        samples = load_samples(os.path.join(manifest_root, "samples.jsonl"))
        by_uid = index_samples_by_uid(samples)
        rho = hparams.get("open_set_ood_ratio", 0.5)
        split_dir = os.path.join(manifest_root, "splits")
        split_map = {
            "A_real_train.json": (ManifestLabeledDataset, label_t),
            "A_painting_train.json": (ManifestLabeledDataset, label_t),
            f"B_mix_train_rho{rho:.2f}.json": (ManifestUnlabeledDataset, (weak_t, strong_t, mask_t)),
            "B_id_eval.json": (ManifestEvalDataset, eval_t),
            "B_ood_eval.json": (ManifestEvalDataset, eval_t),
            "T_clipart_eval.json": (ManifestEvalDataset, eval_t),
        }
        self.datasets = []
        for split_name, (cls, tfm) in split_map.items():
            uids = load_uid_split(os.path.join(split_dir, split_name))
            rows = select_samples(by_uid, uids)
            if cls is ManifestUnlabeledDataset:
                self.datasets.append(cls(rows, weak_transform=tfm[0], strong_transform=tfm[1], mask_transform=tfm[2]))
            else:
                self.datasets.append(cls(rows, transform=tfm))
        self.input_shape = self.INPUT_SHAPE
        return
        self.id_classes = ["book", "clock", "keyboard", "lamp", "mug", "scissors"]
        self.num_classes = len(self.id_classes)
        label_map = {name: idx for idx, name in enumerate(self.id_classes)}

        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        augment_transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.3, 0.3, 0.3, 0.3), transforms.RandomGrayscale(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        required_dirs = [
            os.path.join(root, "domain_net", "real"),
            os.path.join(root, "domain_net", "painting"),
            os.path.join(root, "domain_net", "sketch"),
            os.path.join(root, "domain_net", "clipart"),
        ]
        for d in required_dirs:
            assert os.path.isdir(d), f"Missing required directory: {d}"

        raw_a_real = collect_imagefolder_samples(os.path.join(root, "domain_net", "real"), keep_classes=self.id_classes, label_map=label_map)
        raw_a_painting = collect_imagefolder_samples(os.path.join(root, "domain_net", "painting"), keep_classes=self.id_classes, label_map=label_map)
        raw_b_id = collect_imagefolder_samples(os.path.join(root, "domain_net", "sketch"), keep_classes=self.id_classes, label_map=label_map)
        raw_t_clipart = collect_imagefolder_samples(os.path.join(root, "domain_net", "clipart"), keep_classes=self.id_classes, label_map=label_map)

        assert len(raw_a_real) > 0, "A_real is empty"
        assert len(raw_a_painting) > 0, "A_painting is empty"
        assert len(raw_b_id) > 0, "B_id is empty"
        assert len(raw_t_clipart) > 0, "T_clipart is empty"

        a_real_samples = wrap_samples(raw_a_real, "A_real", "DomainNet", False, "A_real")
        a_painting_samples = wrap_samples(raw_a_painting, "A_painting", "DomainNet", False, "A_painting")
        t_clipart_samples = wrap_samples(raw_t_clipart, "T_clipart", "DomainNet", False, "T_clipart")
        b_id_all = wrap_samples(raw_b_id, "B_id", "DomainNet", False, "B_id")

        terra_samples = []
        terra_root = os.path.join(root, "terra_incognita", "location_100")
        if os.path.isdir(terra_root):
            raw_terra = collect_imagefolder_samples(terra_root)
            terra_samples = wrap_samples([(path, -1) for path, _ in raw_terra], "B_ood", "TerraIncognita", True, "Terra")

        sviro_samples = []
        sviro_root = os.path.join(root, "sviro", "aclass")
        if os.path.isdir(sviro_root):
            raw_sviro = collect_imagefolder_samples(sviro_root)
            sviro_samples = wrap_samples([(path, -1) for path, _ in raw_sviro], "B_ood", "SVIRO", True, "SVIRO")

        spaw_samples = []
        spaw_root = os.path.join(root, "spawrious224")
        if os.path.isdir(spaw_root):
            raw_spaw = []
            for split in sorted(os.listdir(spaw_root)):
                split_root = os.path.join(spaw_root, split)
                if not os.path.isdir(split_root):
                    continue
                for location in sorted(os.listdir(split_root)):
                    location_root = os.path.join(split_root, location)
                    if not os.path.isdir(location_root):
                        continue
                    raw_spaw.extend(collect_imagefolder_samples(location_root))
            spaw_samples = wrap_samples([(path, -1) for path, _ in raw_spaw], "B_ood", "Spawrious224", True, "Spaw")

        max_ood_per_source = hparams.get("open_set_max_ood_per_source", 5000)
        if max_ood_per_source is not None:
            terra_samples = sample_n(terra_samples, min(max_ood_per_source, len(terra_samples)), seed=1)
            sviro_samples = sample_n(sviro_samples, min(max_ood_per_source, len(sviro_samples)), seed=2)
            spaw_samples = sample_n(spaw_samples, min(max_ood_per_source, len(spaw_samples)), seed=3)

        ood_all = terra_samples + sviro_samples + spaw_samples
        assert len(ood_all) > 0, "No OOD samples were found."

        rng = random.Random(0)
        rng.shuffle(b_id_all)
        b_id_split = int(0.8 * len(b_id_all))
        b_id_train = b_id_all[:b_id_split]
        b_id_eval = b_id_all[b_id_split:]
        assert len(b_id_train) > 0, "B_id_train is empty"
        assert len(b_id_eval) > 0, "B_id_eval is empty"

        rng.shuffle(ood_all)
        ood_split = int(0.8 * len(ood_all))
        ood_train_pool = ood_all[:ood_split]
        ood_eval = ood_all[ood_split:]
        assert len(ood_train_pool) > 0, "OOD train pool is empty"
        assert len(ood_eval) > 0, "OOD eval is empty"

        rho = hparams.get("open_set_ood_ratio", 0.5)
        assert 0.0 <= rho < 1.0, f"Invalid open_set_ood_ratio: {rho}"

        num_id = len(b_id_train)
        num_ood = int(num_id * rho / max(1e-8, (1 - rho)))
        ood_selected = sample_n(ood_train_pool, min(num_ood, len(ood_train_pool)), seed=4)
        assert len(ood_selected) > 0, "No OOD selected into B_mix"

        b_mix_samples = b_id_train + ood_selected
        assert len(b_mix_samples) > 0, "B_mix is empty"
        rng.shuffle(b_mix_samples)

        for s in b_id_eval:
            s["env"] = "B_id_eval"
        for s in ood_eval:
            s["env"] = "B_ood_eval"

        self.datasets = [
            SimpleImageListDataset(a_real_samples, transform=augment_transform),
            SimpleImageListDataset(a_painting_samples, transform=augment_transform),
            SimpleImageListDataset(b_mix_samples, transform=transform),
            SimpleImageListDataset(b_id_eval, transform=transform),
            SimpleImageListDataset(ood_eval, transform=transform),
            SimpleImageListDataset(t_clipart_samples, transform=transform),
        ]
        self.input_shape = self.INPUT_SHAPE

        print("[OpenSetDomainNetObjects] ID classes:", self.id_classes)
        print("[OpenSetDomainNetObjects] A_real:", len(a_real_samples))
        print("[OpenSetDomainNetObjects] A_painting:", len(a_painting_samples))
        print("[OpenSetDomainNetObjects] B_id_train:", len(b_id_train))
        print("[OpenSetDomainNetObjects] B_id_eval:", len(b_id_eval))
        print("[OpenSetDomainNetObjects] Terra OOD:", len(terra_samples))
        print("[OpenSetDomainNetObjects] SVIRO OOD:", len(sviro_samples))
        print("[OpenSetDomainNetObjects] Spawrious OOD:", len(spaw_samples))
        print("[OpenSetDomainNetObjects] OOD selected for B_mix:", len(ood_selected))
        print("[OpenSetDomainNetObjects] B_ood_eval:", len(ood_eval))
        print("[OpenSetDomainNetObjects] B_mix:", len(b_mix_samples))
        print("[OpenSetDomainNetObjects] T_clipart:", len(t_clipart_samples))

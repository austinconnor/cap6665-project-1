"""
Inspired from
https://huggingface.co/datasets/ydshieh/coco_dataset_script/blob/main/coco_dataset_script.py
"""

import json
import os
import shutil
import urllib.request
import zipfile
import datasets
import collections


_LOCAL_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_RAW_DATA_DIR = os.path.join(_LOCAL_DATA_DIR, "_raw")
_ZIP_PATH = os.path.join(_RAW_DATA_DIR, "DocLayNet_core.zip")
_EXTRACT_DIR = os.path.join(_RAW_DATA_DIR, "DocLayNet_core")


def _copy_if_needed(src, dst):
    if os.path.exists(dst) and os.path.getsize(src) == os.path.getsize(dst):
        return

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def _materialize_split(archive_root, split):
    json_src = os.path.join(archive_root, "COCO", f"{split}.json")
    image_src_dir = os.path.join(archive_root, "PNG")
    split_dir = os.path.join(_LOCAL_DATA_DIR, split)
    image_dst_dir = os.path.join(split_dir, "PNG")
    json_dst = os.path.join(split_dir, f"{split}.json")

    os.makedirs(image_dst_dir, exist_ok=True)
    _copy_if_needed(json_src, json_dst)

    with open(json_src, encoding="utf8") as f:
        image_infos = json.load(f)["images"]

    for image_info in image_infos:
        image_name = image_info["file_name"]
        _copy_if_needed(
            os.path.join(image_src_dir, image_name),
            os.path.join(image_dst_dir, image_name),
        )

    return json_dst, image_dst_dir


def _download_zip():
    os.makedirs(_RAW_DATA_DIR, exist_ok=True)
    if os.path.exists(_ZIP_PATH):
        print(f"Using existing archive: {_ZIP_PATH}")
        return

    print(f"Downloading {_URLs['core']} to {_ZIP_PATH}")

    def _reporthook(block_count, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = min(block_count * block_size, total_size)
        percent = downloaded * 100 / total_size
        print(f"\rDownloaded {percent:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(_URLs["core"], _ZIP_PATH, _reporthook)
    print()


def _extract_zip():
    if os.path.exists(_EXTRACT_DIR):
        print(f"Using existing extracted data: {_EXTRACT_DIR}")
        return _find_archive_root(_EXTRACT_DIR)

    print(f"Extracting {_ZIP_PATH} to {_EXTRACT_DIR}")
    os.makedirs(_EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(_ZIP_PATH) as zip_file:
        zip_file.extractall(_EXTRACT_DIR)

    return _find_archive_root(_EXTRACT_DIR)


def _find_archive_root(extract_dir):
    if _is_archive_root(extract_dir):
        return extract_dir

    for root, dirs, _files in os.walk(extract_dir):
        if _is_archive_root(root):
            dirs.clear()
            return root

    raise FileNotFoundError(f"Could not find COCO/PNG folders under {extract_dir}")


def _is_archive_root(path):
    return os.path.isdir(os.path.join(path, "COCO")) and os.path.isdir(
        os.path.join(path, "PNG")
    )


def download_to_local_data():
    _download_zip()
    archive_root = _extract_zip()

    for split in ("train", "val", "test"):
        print(f"Creating data/{split}")
        _materialize_split(archive_root, split)

    if os.path.exists(_RAW_DATA_DIR):
        print(f"Removing temporary data: {_RAW_DATA_DIR}")
        shutil.rmtree(_RAW_DATA_DIR)

    print("Done. Split folders are in ./data.")


class COCOBuilderConfig(datasets.BuilderConfig):
    def __init__(self, name, splits, **kwargs):
        super().__init__(name, **kwargs)
        self.splits = splits


# Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@article{doclaynet2022,
  title = {DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis},  
  doi = {10.1145/3534678.353904},
  url = {https://arxiv.org/abs/2206.01062},
  author = {Pfitzmann, Birgit and Auer, Christoph and Dolfi, Michele and Nassar, Ahmed S and Staar, Peter W J},
  year = {2022}
}
"""

# Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
DocLayNet is a human-annotated document layout segmentation dataset from a broad variety of document sources.
"""

# Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://developer.ibm.com/exchanges/data/all/doclaynet/"

# Add the licence for the dataset here if you can find it
_LICENSE = "CDLA-Permissive-1.0"

# Add link to the official dataset URLs here
# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)

_URLs = {
    "core": "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip",
}

# Name of the dataset usually match the script name with CamelCase instead of snake_case
class COCODataset(datasets.GeneratorBasedBuilder):
    """An example dataset script to work with the local (downloaded) COCO dataset"""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = COCOBuilderConfig
    BUILDER_CONFIGS = [
        COCOBuilderConfig(name="2022.08", splits=["train", "val", "test"]),
    ]
    DEFAULT_CONFIG_NAME = "2022.08"

    def _info(self):
        features = datasets.Features(
            {
                "image_id": datasets.Value("int64"),
                "image": datasets.Image(),
                "width": datasets.Value("int32"),
                "height": datasets.Value("int32"),
                # Custom fields
                "doc_category": datasets.Value(
                    "string"
                ),  # high-level document category
                "collection": datasets.Value("string"),  # sub-collection name
                "doc_name": datasets.Value("string"),  # original document filename
                "page_no": datasets.Value("int64"),  # page number in original document
            }
        )
        object_dict = {
            "category_id": datasets.ClassLabel(
                names=[
                    "Caption",
                    "Footnote",
                    "Formula",
                    "List-item",
                    "Page-footer",
                    "Page-header",
                    "Picture",
                    "Section-header",
                    "Table",
                    "Text",
                    "Title",
                ]
            ),
            "image_id": datasets.Value("string"),
            "id": datasets.Value("int64"),
            "area": datasets.Value("int64"),
            "bbox": datasets.Sequence(datasets.Value("float32"), length=4),
            "segmentation": [[datasets.Value("float32")]],
            "iscrowd": datasets.Value("bool"),
            "precedence": datasets.Value("int32"),
        }
        features["objects"] = [object_dict]

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        archive_path = dl_manager.download_and_extract(_URLs)
        splits = []
        for split in self.config.splits:
            if split == "train":
                json_path, image_dir = _materialize_split(archive_path["core"], "train")
                dataset = datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "json_path": json_path,
                        "image_dir": image_dir,
                        "split": "train",
                    },
                )
            elif split in ["val", "valid", "validation", "dev"]:
                json_path, image_dir = _materialize_split(archive_path["core"], "val")
                dataset = datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "json_path": json_path,
                        "image_dir": image_dir,
                        "split": "val",
                    },
                )
            elif split == "test":
                json_path, image_dir = _materialize_split(archive_path["core"], "test")
                dataset = datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "json_path": json_path,
                        "image_dir": image_dir,
                        "split": "test",
                    },
                )
            else:
                continue

            splits.append(dataset)
        return splits

    def _generate_examples(
        # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
        self,
        json_path,
        image_dir,
        split,
    ):
        """Yields examples as (key, example) tuples."""
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.
        def _image_info_to_example(image_info, image_dir):
            image = image_info["file_name"]
            return {
                "image_id": image_info["id"],
                "image": os.path.join(image_dir, image),
                "width": image_info["width"],
                "height": image_info["height"],
                "doc_category": image_info["doc_category"],
                "collection": image_info["collection"],
                "doc_name": image_info["doc_name"],
                "page_no": image_info["page_no"],
            }

        with open(json_path, encoding="utf8") as f:
            annotation_data = json.load(f)
            images = annotation_data["images"]
            annotations = annotation_data["annotations"]
            image_id_to_annotations = collections.defaultdict(list)
            for annotation in annotations:
                image_id_to_annotations[annotation["image_id"]].append(annotation)

        for idx, image_info in enumerate(images):
            example = _image_info_to_example(image_info, image_dir)
            annotations = image_id_to_annotations[image_info["id"]]
            objects = []
            for annotation in annotations:
                category_id = annotation["category_id"]  # Zero based counting
                if category_id != -1:
                    category_id = category_id - 1
                annotation["category_id"] = category_id
                objects.append(annotation)
            example["objects"] = objects
            yield idx, example


if __name__ == "__main__":
    download_to_local_data()
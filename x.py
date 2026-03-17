# # import mlcroissant as mlc

# # # 1. Define the Distribution (FileSets)
# # distribution: list[mlc.FileSet | mlc.FileObject] = [
# #     mlc.FileSet(
# #         id="mirax-files",
# #         name="MIRAX Index Files",
# #         includes=["*.mrxs", "**/*.dat"],
# #         encoding_formats=["application/octet-stream"],
# #     ),
# #     mlc.FileSet(
# #         id="annotation-files",
# #         name="XML Annotations",
# #         includes=["*.xml"],
# #         encoding_formats=["text/xml"],
# #     ),
# # ]

# # # 2. Define the RecordSets
# # record_sets = [
# #     # Slide RecordSet: Extracts the ID from the filename
# #     mlc.RecordSet(
# #         id="slides",
# #         fields=[
# #             mlc.Field(
# #                 id="slides/file_path",
# #                 data_types=[mlc.DataType.TEXT],
# #                 source=mlc.Source(
# #                     file_set="mirax-files",
# #                     extract=mlc.Extract(file_property=mlc.FileProperty.filepath),
# #                 ),
# #             ),
# #         ],
# #     ),
# #     # Annotations RecordSet: Links to Slides via regex
# #     mlc.RecordSet(
# #         id="annotations",
# #         fields=[
# #             mlc.Field(
# #                 id="annotations/slide_ref",
# #                 references=mlc.Source(field="slides/slide_id"),
# #                 source=mlc.Source(
# #                     file_set="annotation-files",
# #                     extract=mlc.Extract(file_property=mlc.FileProperty.filename),
# #                     transforms=[mlc.Transform(regex=r"(.*)\.xml")],
# #                 ),
# #             ),
# #             mlc.Field(
# #                 id="annotations/label_content",
# #                 data_types=[mlc.DataType.TEXT],
# #                 source=mlc.Source(
# #                     file_set="annotation-files",
# #                     extract=mlc.Extract(file_property=mlc.FileProperty.content),
# #                 ),
# #             ),
# #         ],
# #     ),
# # ]

# # # 3. Create the Metadata object
# # metadata = mlc.Metadata(
# #     name="my-mirax-dataset",
# #     description="WSI MIRAX dataset with XML annotations.",
# #     distribution=distribution,
# #     record_sets=record_sets,
# #     url="https://example.com/dataset",
# # )

# # # 4. Generate the JSON-LD
# # print(metadata.to_json())


# # data = mlc.Dataset()

# # data.records("annotations")


import mlcroissant as mlc

# 1. Distribution: Use globs that capture the directory structure
distribution = [
    mlc.FileSet(
        id="mirax-files",
        name="MIRAX Index Files",
        includes=["**/*.mrxs", "**/*.dat"],
        encoding_format="application/octet-stream",
    ),
    mlc.FileSet(
        id="annotation-files",
        name="XML Annotations",
        includes=["**/*.xml"],
        encoding_format="text/xml",
    ),
    mlc.FileObject(),
]

# 2. Define the RecordSets
record_sets = [
    # A. Define the Splits Enumeration
    mlc.RecordSet(
        id="splits",
        data_types=[mlc.DataType.SPLIT],
        description="Dataset splits (train/test)",
        data=[
            {"name": "train", "split": "cr:TrainingSplit"},
            {"name": "test", "split": "cr:TestSplit"},
        ],
        fields=[
            mlc.Field(
                id="splits/name",
                data_types=[mlc.DataType.TEXT],
                source=mlc.Source(extract=mlc.Extract(column="name")),
            ),
            mlc.Field(
                id="splits/split",
                data_types=[mlc.DataType.TEXT],
                source=mlc.Source(extract=mlc.Extract(column="split")),
            ),
        ],
    ),
    # B. Slides RecordSet
    mlc.RecordSet(
        id="slides",
        fields=[
            # ID: Extract filename without extension for joining
            mlc.Field(
                id="slides/slide_id",
                data_types=[mlc.DataType.TEXT],
                source=mlc.Source(
                    file_set="mirax-files",
                    extract=mlc.Extract(file_property="filename"),
                    transforms=[mlc.Transform(regex=r"(.*)\.mrxs")],
                ),
            ),
            # Split: Determine if it's in train/ or test/ folder
            mlc.Field(
                id="slides/split",
                references=mlc.Source(field="splits/name"),
                source=mlc.Source(
                    file_set="mirax-files",
                    extract=mlc.Extract(file_property="fullpath"),
                    transforms=[mlc.Transform(regex=r"(train|test)/slides/.*")],
                ),
            ),
            mlc.Field(
                id="slides/file_path",
                data_types=[mlc.DataType.TEXT],
                source=mlc.Source(
                    file_set="mirax-files",
                    extract=mlc.Extract(file_property="fullpath"),
                ),
            ),
        ],
    ),
    # C. Annotations RecordSet
    mlc.RecordSet(
        id="annotations",
        fields=[
            # Join key: Extract ID from the XML filename
            mlc.Field(
                id="annotations/slide_ref",
                references=mlc.Source(field="slides/slide_id"),
                source=mlc.Source(
                    file_set="annotation-files",
                    extract=mlc.Extract(file_property="filename"),
                    transforms=[mlc.Transform(regex=r"(.*)\.xml")],
                ),
            ),
            mlc.Field(
                id="annotations/label_content",
                data_types=[mlc.DataType.TEXT, "cr:Label"],
                source=mlc.Source(
                    file_set="annotation-files",
                    extract=mlc.Extract(file_property="content"),
                ),
            ),
        ],
    ),
]


# mlc.Dataset("").records("train")

metadata = mlc.Metadata(
    name="my-mirax-dataset",
    description="WSI MIRAX dataset with XML annotations.",
    distribution=distribution,
    record_sets=record_sets,
    url="https://example.com/dataset",
)


from mlcroissant import Dataset

ds = Dataset(jsonld="https://huggingface.co/api/datasets/RationAI/PanNuke/croissant")
records = ds.records("default")
for record in records:
    print(record)
    break


from mlcroissant import Dataset

# The Croissant metadata exposes the first 5GB of this dataset
ds = Dataset(
    jsonld="https://huggingface.co/api/datasets/lance-format/Openvid-1M/croissant"
)
records = ds.records("default")


for r in records:
    print(r)
    break


import mlcroissant as mlc

ta_factory = mlc.torch.LoaderFactory(
    "https://huggingface.co/api/datasets/lance-format/Openvid-1M/croissant"
)
ta_factory.as_datapipe("default")

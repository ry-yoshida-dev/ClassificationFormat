# classification_format

## Overview

Format-independent classification annotation model and writers serializing it into dataset formats.

## Components

| Component | Description |
| --------- | ----------- |
| [annotation/](./annotation/README.md) | Validated labels, images and the dataset with its class list and label cardinality. |
| [writer/](./writer/README.md) | Writers for ImageFolder, CSV, Hugging Face, Pascal VOC and Open Images, and the `AnnotationFormat` enum. |

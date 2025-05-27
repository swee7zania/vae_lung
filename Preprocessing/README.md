# `Train_Test_Split.ipynb`

1. **Read and preprocess data:**

   - Load the metadata file `meta_info.csv`;
   - Delete irrelevant columns (such as `is_clean`);
   - Filter out nodule-related samples by judging whether the characters in the original_image column (positions 6–7 are "NI").

   | Slices | Patient |
   | ------ | ------- |
   | 13607  | 862     |

2. **Data Filtering:**

   - Removed samples with malignancy == 3 (Ambiguous).
   - Statistics on the distribution of malignancy and is_cancer for the remaining data.

   | malignancy    | 3             | 5307 (Deleted)     |
   | ------------- | ------------- | ------------------ |
   |               | 1             | 1426               |
   |               | 2             | 1800               |
   |               | 4             | 2448               |
   |               | 5             | 2626               |
   | **is_cancer** | **Ambiguous** | **5307 (Deleted)** |
   |               | True          | 5074               |
   |               | False         | 3226               |

   | Slices | Patient |
   | ------ | ------- |
   | 8300   | 669     |

3. **Splitting the dataset:**

   - Split the dataset based on patient ID;
   - Divide patients into 80% training + 20% testing;
   - Take 13% from the training set for validation;
   - Use the is_train function to assign labels (Train, Validation, Test) to each data sample.

   |              | Train | Test | Validation | Total |
   | ------------ | ----- | ---- | ---------- | ----- |
   | **Patients** | 465   | 134  | 70         | 669   |
   | **Slices**   | 5738  | 1649 | 913        | 8300  |

4. **Filter the data:**

   - Only keep the records of image files in the directory.

5. **Save label data:**

   - Count the data of cancer and no cancer;
   - Record the label of cancer as 1 and no cancer as 2.

   |              | **cancer** | **noncancer** | **Total** |
   | ------------ | ---------- | ------------- | --------- |
   | **Patients** | 383        | 286           | 669       |
   | **Slices**   | 5074       | 3216          | 8290      |



# Metadata

| Field Name       | Type    | Description                                                  |
| ---------------- | ------- | ------------------------------------------------------------ |
| `index`          | int     | Unique row index.                                            |
| `patient_id`     | int     | Identifier for the patient.                                  |
| `nodule_no`      | int     | Nodule index within a patient.                               |
| `slice_no`       | int     | CT scan slice number.                                        |
| `original_image` | string  | File name of the original CT image (e.g., `0001_NI000_slice000`). |
| `mask_image`     | string  | File name of the corresponding segmentation mask image (e.g., `0001_MA000_slice000`). |
| `malignancy`     | int     | Malignancy score ranging typically from 1 to 5, where 5 indicates high malignancy. |
| `is_cancer`      | boolean | Indicates whether the nodule is cancerous (`TRUE` for cancer, `FALSE` for no cancer). |
| `is_nodule`      | boolean | Indicates whether the region is a nodule (`TRUE` if it's a nodule, `FALSE` if it's not a nodule). |
| `data_split`     | string  | Dataset split information, specifying whether the sample is used for Train, Validation or Test. |
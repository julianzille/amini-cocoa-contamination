# Class labels and mapping
ID2LABEL = {2: "healthy", 1: "cssvd", 0: "anthracnose"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# Default model
BASE_MODEL = "PekingU/rtdetr_v2_r50vd"
DATASET_REPO = "julianz1/cocoa-disease-detection"

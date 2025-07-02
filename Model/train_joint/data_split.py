import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

def load_meta_and_images(meta_path, image_dir, k_folds=5, random_state=42):
    """
    加载meta数据并筛选data_split为'Train'的样本，然后进行K折划分。

    Args:
        meta_path (str): meta csv
        image_dir (str): 图像目录路径
        k_folds (int): K折交叉验证的折数
        random_state (int): 随机种子

    Returns:
        List[Tuple[train_df, val_df]]: 每折的训练与验证数据列表
    """
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    if not os.path.exists(image_dir):
        print(f"⚠️ Warning: Image directory not found: {image_dir} (will still proceed)")

    meta_df = pd.read_csv(meta_path)

    required_cols = {"original_image", "data_split", "label"}
    if not required_cols.issubset(meta_df.columns):
        missing = required_cols - set(meta_df.columns)
        raise ValueError(f"Missing required columns in meta file: {missing}")

    # # 只选出 Train 样本
    # meta_df = meta_df[meta_df["data_split"] == "Train"].copy()
    # if meta_df.empty:
    #     raise ValueError("No samples with data_split == 'Train' found in meta file.")

    meta_df = shuffle(meta_df, random_state=random_state).reset_index(drop=True)

    # 构造图像路径列
    meta_df["image_path"] = meta_df["original_image"].apply(lambda x: os.path.join(image_dir, x + ".npy"))

    # 创建 K 折划分
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    splits = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(meta_df, meta_df["label"])):
        train_df = meta_df.iloc[train_idx].reset_index(drop=True)
        val_df = meta_df.iloc[val_idx].reset_index(drop=True)
        # print(f"Fold {fold + 1}: Train size = {len(train_df)}, Val size = {len(val_df)}")
        splits.append((train_df, val_df))
    return splits

# 示例调用（可删除）
if __name__ == "__main__":
    image_dir = r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Images"
    meta_file = r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Meta/meta_mal_ben.csv"
    splits = load_meta_and_images(meta_file, image_dir, k_folds=5)

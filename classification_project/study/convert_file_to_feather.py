import pandas as pd
# from classification_project.preprocessing.preprocessing import Preprocessing

# csv_df_path = '../../data/processed/cifar-10-100-augmentation.csv'
# df = pd.read_csv(csv_df_path)
# feather_df_path = '../../data/processed/cifar_10_100_augmentation.feather'
# df.to_feather(feather_df_path)

df_path = '../../data/processed/cifar_10_100_augmentation.feather'
df = pd.read_feather(df_path)

df.iloc[:, 2:] = df.iloc[:, 2:].astype('float32') / 255

transform_df_path = '../../data/processed/df_cifar_10_100_augmentation_normalized.feather'
df.to_feather(transform_df_path)


# preprocessing = Preprocessing(df)
# transform_df = preprocessing.normalize_and_scale_data(return_data=True)
#
# transform_df_path = '../../data/processed/df_cifar_10_100_augmentation_normalized_and_scaled.feather'
# transform_df.to_feather(transform_df_path)

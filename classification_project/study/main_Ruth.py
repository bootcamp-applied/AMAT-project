import pandas as pd
from classification_project.study.use_Visualization import Use_Visualization

if __name__ == '__main__':

    df = pd.read_csv('../../data/processed/cifar-10-100.csv')
    #pareto to df
    Use_Visualization.pareto_to_df_label(df)
    #pareto to train_val_test
    Use_Visualization.pareto_tarin_val_test(df)



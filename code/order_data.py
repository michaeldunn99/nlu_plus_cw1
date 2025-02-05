import pandas as pd

def main():
    
    file_path = "output/parsed_results.csv"
    df= pd.read_csv(file_path)
    column_to_sort = "best_loss"
    df_sorted = df.sort_values(by=column_to_sort, ascending=True)
    print(df_sorted.head())

    df_sorted.to_csv("output/sorted_parsed_results.csv", index=False)

if __name__ == "__main__":
    main()


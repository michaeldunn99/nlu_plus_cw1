import re
import pandas as pd

def parse_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Regular expressions for extracting values, updated to handle large gaps
    pattern = re.compile(
        r"hdim = (?P<hidden_units>\d+)[\s\S]*?"
        r"Vocab size: (?P<vocab_size>\d+)[\s\S]*?"
        r"Hidden units: (?P=hidden_units)[\s\S]*?"
        r"Steps for back propagation: (?P<steps_bp>\d+)[\s\S]*?"
        r"Initial learning rate set to (?P<learning_rate>[\d\.]+)[\s\S]*?"
        r"annealing set to (?P<annealing_rate>[\d\.]+)[\s\S]*?"
        r"best observed loss was (?P<best_loss>[\d\.]+)", 
        re.DOTALL
    )
    
    results = []
    for match in pattern.finditer(text):
        results.append(match.groupdict())
    
    # Convert results to a DataFrame
    df = pd.DataFrame(results)
    df = df.astype({"hidden_units": int, "vocab_size": int, "steps_bp": int, "learning_rate": float, "annealing_rate": float, "best_loss": float})
    
    return df

def main():
    # Example usage
    file_path = "output/hyper_param_tuning_1000.txt"  # Update this path
    results_df = parse_results(file_path)
    results_df.to_csv("output/parsed_results.csv", index=False)
    print("Results saved to parsed_results.csv")

if __name__ == "__main__":
    main()

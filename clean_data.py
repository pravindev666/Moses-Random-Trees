import os

def clean_csv(filepath):
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    # Set to keep track of rows we've already seen to prevent duplicates from the merge zones
    seen_dates = set()
    header = lines[0]
    cleaned_lines.append(header)
    
    for line in lines[1:]:
        # Skip merge markers
        if any(marker in line for marker in ["<<<<<<<", "=======", ">>>>>>>"]):
            continue
        
        # Check for duplicates (based on date/time which is the first column)
        parts = line.split(',')
        if len(parts) > 0:
            date_val = parts[0].strip()
            if date_val in seen_dates:
                continue
            seen_dates.add(date_val)
            cleaned_lines.append(line)
            
    with open(filepath, 'w') as f:
        f.writelines(cleaned_lines)
    print(f"Cleaned {filepath}. Kept {len(cleaned_lines)} lines.")

# Clean the files
DATA_DIR = r"c:\Users\hp\Desktop\New_ML\Moses-RandomForest\data"
clean_csv(os.path.join(DATA_DIR, "INDIAVIX_15minute_2001_now.csv"))
clean_csv(os.path.join(DATA_DIR, "nifty_15m_2001_to_now.csv"))

import os
import pandas as pd

def pe_valuation(eps, target_pe=13.3):
    """
    Calculate valuation based on P/E ratio.
    
    Parameters:
        eps (float): Earnings per share
        target_pe (float): Benchmark P/E ratio
        
    Returns:
        float: Intrinsic value
    """
    return eps * target_pe

def pb_valuation(bvps, target_pb=1.46):
    """
    Calculate valuation based on P/B ratio.
    
    Parameters:
        bvps (float): Book value per share
        target_pb (float): Benchmark P/B ratio
        
    Returns:
        float: Intrinsic value
    """
    return bvps * target_pb

def graham_simplified_valuation(eps, g=10):
    """
    Simplified Graham formula valuation.
    
    V = EPS × (7 + 1.5g)
    
    Parameters:
        eps (float): Earnings per share
        g (float): Growth rate (e.g., 10 for 10%)
        
    Returns:
        float: Intrinsic value
    """
    return eps * (7 + 1.5 * g)

def graham_full_valuation(eps, g=10, y=6.5):
    """
    Full Graham formula valuation.
    
    V = (EPS × (7 + 1.5g) × 4.4) / y
    
    Parameters:
        eps (float): Earnings per share
        g (float): Growth rate (e.g., 10 for 10%)
        y (float): Bond yield (e.g., 6.5 for 6.5%)
        
    Returns:
        float: Intrinsic value
    """
    return (eps * (7 + 1.5 * g) * 4.4) / y if y != 0 else float('inf')

# Handle millisecond timestamps (if any)
def convert_if_millis(date_str):
    try:
        if date_str.isdigit() and len(date_str) >= 13:
            dt = pd.to_datetime(int(date_str), unit='ms')
        else:
            dt = pd.to_datetime(date_str, utc=True, errors='coerce')  # force UTC

        return dt.tz_localize(None) if dt.tzinfo is not None else dt
    except:
        return pd.NaT

def save_dataframe_to_csv(df: pd.DataFrame, filename: str):
    """
    Save DataFrame to a CSV file.
    """
    if df.empty:
        print("[WARN] No data to save.")
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"[SUCCESS] Data saved to: {filename}")

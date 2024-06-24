import pandas as pd
import numpy as np
import random

# Define the number of records
num_records = 1000

# Generate synthetic data
data = {
    'Duration': np.random.randint(0, 100, num_records),
    'Protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], num_records),
    'Service': np.random.choice(['HTTP', 'FTP', 'SMTP', 'DNS'], num_records),
    'Flag': np.random.choice(['SF', 'REJ', 'RSTO', 'S0'], num_records),
    'Src_Bytes': np.random.randint(0, 10000, num_records),
    'Dst_Bytes': np.random.randint(0, 10000, num_records),
    'Failed_Logins': np.random.randint(0, 5, num_records),
    'Root_Accesses': np.random.randint(0, 5, num_records),
    'File_Creations': np.random.randint(0, 5, num_records),
    'Shell_Prompts': np.random.randint(0, 5, num_records),
    'Same_Host_Conn': np.random.randint(0, 20, num_records),
    'Same_Service_Conn': np.random.randint(0, 20, num_records),
    'Src_To_Dst_Pkts': np.random.randint(0, 100, num_records),
    'Dst_To_Src_Pkts': np.random.randint(0, 100, num_records),
    'Avg_Pkt_Size': np.random.randint(0, 1500, num_records),
    'Label': np.random.choice(['Normal', 'DoS', 'Probe', 'R2L', 'U2R'], num_records)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('network_security_data.csv', index=False)


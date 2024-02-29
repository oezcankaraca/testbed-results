import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data from CSV file
df = pd.read_csv('/Users/ozcankaraca/Desktop/Testresults2/AllresultsLast.csv', sep=";")

df.columns = df.columns.str.strip()
del df["TestID"]

#------------------------------------------------------------------------------------------------------------------
# 1 - Error rate for bandwidth with an increasing number of peers (with P2P) for a specific file size.

df['P2P Algorithm'] = df['P2P Algorithm'].astype(str)

df_filtered = df[(df['P2P Algorithm'].str.upper() == 'USED') & (df['Total Received Bytes'] == 67108864)]

grouped_data = df_filtered.groupby('Number of Peers')[['Maximum Bandwidth Error Rate [%]', 
                                                       'Minimum Bandwidth Error Rate [%]', 
                                                       'Average Bandwidth Error Rate [%]']].mean().reset_index()

plt.figure(figsize=(10, 6), dpi=300) 

plt.plot(grouped_data['Number of Peers'], grouped_data['Maximum Bandwidth Error Rate [%]'], 
         label='Maximum', marker='o', linestyle='-', linewidth=2)
plt.plot(grouped_data['Number of Peers'], grouped_data['Minimum Bandwidth Error Rate [%]'], 
         label='Minimum', marker='s', linestyle='-', linewidth=2)
plt.plot(grouped_data['Number of Peers'], grouped_data['Average Bandwidth Error Rate [%]'], 
         label='Average', marker='^', linestyle='-', linewidth=2)

plt.xlabel('Number of Peers', fontsize=20)
plt.ylabel('Bandwidth Error Rate [%]', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16) 

plt.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('bandwidth_error_rate_with_p2p.png', format='png', dpi=300) 
plt.show()

#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
# 2 - Error rate for bandwidth with an increasing number of peers (without P2P) for a specific file size

df['P2P Algorithm'] = df['P2P Algorithm'].astype(str)

df_filtered = df[(df['P2P Algorithm'].str.upper() == 'NOT USED') & (df['Total Received Bytes'] == 67108864)]

grouped_data = df_filtered.groupby('Number of Peers')[['Maximum Bandwidth Error Rate [%]', 
                                                       'Minimum Bandwidth Error Rate [%]', 
                                                       'Average Bandwidth Error Rate [%]']].mean().reset_index()

plt.figure(figsize=(10, 6), dpi=300)

plt.plot(grouped_data['Number of Peers'], grouped_data['Maximum Bandwidth Error Rate [%]'], 
         label='Maximum', marker='o', linestyle='-', linewidth=2)
plt.plot(grouped_data['Number of Peers'], grouped_data['Minimum Bandwidth Error Rate [%]'], 
         label='Minimum', marker='s', linestyle='-', linewidth=2)
plt.plot(grouped_data['Number of Peers'], grouped_data['Average Bandwidth Error Rate [%]'], 
         label='Average', marker='^', linestyle='-', linewidth=2)

plt.xlabel('Number of Peers', fontsize=20)
plt.ylabel('Bandwidth Error Rate [%]', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16) 

plt.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('bandwidth_error_rate_without_p2p.png', format='png', dpi=300)
plt.show()

#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
# 3 - Error rate for latency with an increasing number of peers (with P2P) for a specific file size.

df['P2P Algorithm'] = df['P2P Algorithm'].astype(str)

df_filtered = df[(df['P2P Algorithm'].str.upper() == 'USED') & (df['Total Received Bytes'] == 67108864)]

grouped_data = df_filtered.groupby('Number of Peers')[['Maximum Latency Error Rate [%]', 
                                                       'Minimum Latency Error Rate [%]', 
                                                       'Average Latency Error Rate [%]']].mean().reset_index()

plt.figure(figsize=(10, 6), dpi=300)

plt.plot(grouped_data['Number of Peers'], grouped_data['Maximum Latency Error Rate [%]'], 
         label='Maximum', marker='o', linestyle='-', linewidth=2)
plt.plot(grouped_data['Number of Peers'], grouped_data['Minimum Latency Error Rate [%]'], 
         label='Minimum', marker='s', linestyle='-', linewidth=2)
plt.plot(grouped_data['Number of Peers'], grouped_data['Average Latency Error Rate [%]'], 
         label='Average', marker='^', linestyle='-', linewidth=2)

plt.xlabel('Number of Peers', fontsize=20)
plt.ylabel('Latency Error Rate [%]', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16) 

plt.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('latency_error_rate_with_p2p.png', format='png', dpi=300)
plt.show()

#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
# 4 - Error rate for latency with an increasing number of peers (without P2P) for a specific file size.

df['P2P Algorithm'] = df['P2P Algorithm'].astype(str)

df_filtered = df[(df['P2P Algorithm'].str.upper() == 'NOT USED') & (df['Total Received Bytes'] == 67108864)]

grouped_data = df_filtered.groupby('Number of Peers')[['Maximum Latency Error Rate [%]', 
                                                       'Minimum Latency Error Rate [%]', 
                                                       'Average Latency Error Rate [%]']].mean().reset_index()

plt.figure(figsize=(10, 6), dpi=300)

plt.plot(grouped_data['Number of Peers'], grouped_data['Maximum Latency Error Rate [%]'],
         label='Maximum', marker='o', linestyle='-', linewidth=2)
plt.plot(grouped_data['Number of Peers'], grouped_data['Minimum Latency Error Rate [%]'],
         label='Minimum', marker='s', linestyle='-', linewidth=2)
plt.plot(grouped_data['Number of Peers'], grouped_data['Average Latency Error Rate [%]'],
         label='Average', marker='^', linestyle='-', linewidth=2)

plt.xlabel('Number of Peers', fontsize=20)
plt.ylabel('Latency Error Rate [%]', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16) 

plt.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('latency_error_rate_without_p2p.png', format='png', dpi=300)
plt.show()

#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
# 5 - Connection time for a connection between two container with an increasing number of peers (with P2P) for a specific file size.

df['P2P Algorithm'] = df['P2P Algorithm'].astype(str)

df_filtered = df[(df['P2P Algorithm'].str.upper() == 'USED') & (df['Total Received Bytes'] == 2239815)]

grouped_data = df_filtered.groupby('Number of Peers')[['Maximum Connection Time [s]', 
                                                       'Minimum Connection Time [s]', 
                                                       'Average Connection Time [s]']].mean().reset_index()

plt.figure(figsize=(10, 6), dpi=300)

plt.plot(grouped_data['Number of Peers'], grouped_data['Maximum Connection Time [s]'],
         label='Max', marker='o', linestyle='-', linewidth=2)
plt.plot(grouped_data['Number of Peers'], grouped_data['Minimum Connection Time [s]'],
         label='Min', marker='s', linestyle='-', linewidth=2)
plt.plot(grouped_data['Number of Peers'], grouped_data['Average Connection Time [s]'],
         label='Avg ', marker='^', linestyle='-', linewidth=2)

plt.xlabel('Number of Peers', fontsize=20)
plt.ylabel('Connection Time [s]', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16) 

plt.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('connection_time_increasing_number_peers_with_p2p.png', format='png', dpi=300)
plt.show()

#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
# 6 - Connection time for a connection between two container with an increasing number of peers (without P2P) for a specific file size.

df['P2P Algorithm'] = df['P2P Algorithm'].astype(str)

df_filtered = df[(df['P2P Algorithm'].str.upper() == 'NOT USED') & (df['Total Received Bytes'] == 2239815)]

grouped_data = df_filtered.groupby('Number of Peers')[['Maximum Connection Time [s]', 
                                                       'Minimum Connection Time [s]', 
                                                       'Average Connection Time [s]']].mean().reset_index()

plt.figure(figsize=(10, 6), dpi=300)

plt.plot(grouped_data['Number of Peers'], grouped_data['Maximum Connection Time [s]'],
         label='Max', marker='o', linestyle='-', linewidth=2)
plt.plot(grouped_data['Number of Peers'], grouped_data['Minimum Connection Time [s]'],
         label='Min', marker='s', linestyle='-', linewidth=2)
plt.plot(grouped_data['Number of Peers'], grouped_data['Average Connection Time [s]'],
         label='Avg', marker='^', linestyle='-', linewidth=2)

plt.xlabel('Number of Peers', fontsize=20)
plt.ylabel('Connection Time [s]', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16) 

plt.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('connection_time_increasing_number_peers_without_p2p.png', format='png', dpi=300)
plt.show()

#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
# 7 - Transfer time of two container with an increasing number of peers (with P2P) for a specific file size.

df['P2P Algorithm'] = df['P2P Algorithm'].astype(str)

df_filtered = df[(df['P2P Algorithm'].str.upper() == 'USED') & (df['Total Received Bytes'] == 15890720)]

grouped_data = df_filtered.groupby('Number of Peers')[['Maximum Transfer Time [s]', 
                                                       'Minimum Transfer Time [s]', 
                                                       'Average Transfer Time [s]']].mean().reset_index()

plt.figure(figsize=(10, 6), dpi=300)

plt.plot(grouped_data['Number of Peers'], grouped_data['Maximum Transfer Time [s]'],
         label='Max', marker='o', linestyle='-', linewidth=2)
plt.plot(grouped_data['Number of Peers'], grouped_data['Minimum Transfer Time [s]'],
         label='Min', marker='s', linestyle='-', linewidth=2)
plt.plot(grouped_data['Number of Peers'], grouped_data['Average Transfer Time [s]'],
         label='Avg', marker='^', linestyle='-', linewidth=2)

plt.xlabel('Number of Peers', fontsize=20)
plt.ylabel('Transfer Time [s]', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16) 

plt.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('transfer_time_increasing_number_peers_with_p2p.png', format='png', dpi=300)
plt.show()

#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
# 8 - Transfer time of two container with an increasing number of peers (without P2P) for a specific file size.

df['P2P Algorithm'] = df['P2P Algorithm'].astype(str)

df_filtered = df[(df['P2P Algorithm'].str.upper() == 'NOT USED') & (df['Total Received Bytes'] == 15890720)]

grouped_data = df_filtered.groupby('Number of Peers')[['Maximum Transfer Time [s]', 
                                                       'Minimum Transfer Time [s]', 
                                                       'Average Transfer Time [s]']].mean().reset_index()

plt.figure(figsize=(10, 6), dpi=300)

plt.plot(grouped_data['Number of Peers'], grouped_data['Maximum Transfer Time [s]'],
         label='Max', marker='o', linestyle='-', linewidth=2)
plt.plot(grouped_data['Number of Peers'], grouped_data['Minimum Transfer Time [s]'],
         label='Min', marker='s', linestyle='-', linewidth=2)
plt.plot(grouped_data['Number of Peers'], grouped_data['Average Transfer Time [s]'],
         label='Avg', marker='^', linestyle='-', linewidth=2)

plt.xlabel('Number of Peers', fontsize=20)
plt.ylabel('Transfer Time [s]', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16) 

plt.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('transfer_time_increasing_number_peers_without_p2p.png', format='png', dpi=300)
plt.show()

#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
# 9 - Total time of two container with an increasing number of peers (with P2P) for a specific file size.

df['P2P Algorithm'] = df['P2P Algorithm'].astype(str)

df_filtered = df[(df['P2P Algorithm'].str.upper() == 'USED') & (df['Total Received Bytes'] == 15890720)]

grouped_data = df_filtered.groupby('Number of Peers')[['Maximum Total Time [s]', 
                                                       'Minimum Total Time [s]', 
                                                       'Average Total Time [s]']].mean().reset_index()

plt.figure(figsize=(10, 6), dpi=300)

plt.plot(grouped_data['Number of Peers'], grouped_data['Maximum Total Time [s]'],
         label='Maximum', marker='o', linestyle='-', linewidth=2)
plt.plot(grouped_data['Number of Peers'], grouped_data['Minimum Total Time [s]'],
         label='Minimum', marker='s', linestyle='-', linewidth=2)
plt.plot(grouped_data['Number of Peers'], grouped_data['Average Total Time [s]'],
         label='Average', marker='^', linestyle='-', linewidth=2)

plt.xlabel('Number of Peers', fontsize=20)
plt.ylabel('Total Time [s]', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16) 

plt.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('total_time_increasing_number_peers_with_p2p.png', format='png', dpi=300)
plt.show()

#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
# 10 - Total time of two container with an increasing number of peers (without P2P) for a specific file size.

df['P2P Algorithm'] = df['P2P Algorithm'].astype(str)

df_filtered = df[(df['P2P Algorithm'].str.upper() == 'NOT USED') & (df['Total Received Bytes'] == 15890720)]

grouped_data = df_filtered.groupby('Number of Peers')[['Maximum Total Time [s]', 
                                                       'Minimum Total Time [s]', 
                                                       'Average Total Time [s]']].mean().reset_index()

plt.figure(figsize=(10, 6), dpi=300)

plt.plot(grouped_data['Number of Peers'], grouped_data['Maximum Total Time [s]'],
         label='Maximum', marker='o', linestyle='-', linewidth=2)
plt.plot(grouped_data['Number of Peers'], grouped_data['Minimum Total Time [s]'],
         label='Minimum', marker='s', linestyle='-', linewidth=2)
plt.plot(grouped_data['Number of Peers'], grouped_data['Average Total Time [s]'],
         label='Average', marker='^', linestyle='-', linewidth=2)

plt.xlabel('Number of Peers', fontsize=20)
plt.ylabel('Total Time [s]', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16) 

plt.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('total_time_increasing_number_peers_without_p2p.png', format='png', dpi=300)
plt.show()

#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
# 11 - Total duration of the whole process with 20 peers and an increasing size of file with and without P2P

df['P2P Algorithm'] = df['P2P Algorithm'].astype(str)

df_with_super_peer = df[(df['P2P Algorithm'].str.upper() == 'USED') & (df['Number of Peers'] == 20)]

df_without_super_peer = df[(df['P2P Algorithm'].str.upper() == 'NOT USED') & (df['Number of Peers'] == 20)]

grouped_data_with = df_with_super_peer.groupby('Total Received Bytes')['Total Duration [s]'].mean().reset_index()
grouped_data_without = df_without_super_peer.groupby('Total Received Bytes')['Total Duration [s]'].mean().reset_index()

plt.figure(figsize=(12, 7), dpi=300)

plt.plot(grouped_data_with['Total Received Bytes'], grouped_data_with['Total Duration [s]'],
         label='With P2P', marker='o', linestyle='-', linewidth=2)
plt.plot(grouped_data_without['Total Received Bytes'], grouped_data_without['Total Duration [s]'],
         label='Without P2P', marker='s', linestyle='-', linewidth=2)

plt.xlabel('File Size (Total Received Bytes)', fontsize=20)
plt.ylabel('Total Duration [s]', fontsize=20)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16) 

plt.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('total_duration_with_20_peers_with_and_without_p2p.png', format='png', dpi=300)
plt.show()

#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
# 12 - Total duration of the whole process with 35 peers and an increasing size of file with and without P2P

df['P2P Algorithm'] = df['P2P Algorithm'].astype(str)

df_with_super_peer = df[(df['P2P Algorithm'].str.upper() == 'USED') & (df['Number of Peers'] == 10)]

df_without_super_peer = df[(df['P2P Algorithm'].str.upper() == 'NOT USED') & (df['Number of Peers'] == 10)]

grouped_data_with = df_with_super_peer.groupby('Total Received Bytes')['Total Duration [s]'].mean().reset_index()
grouped_data_without = df_without_super_peer.groupby('Total Received Bytes')['Total Duration [s]'].mean().reset_index()

plt.figure(figsize=(12, 7), dpi=300)

plt.plot(grouped_data_with['Total Received Bytes'], grouped_data_with['Total Duration [s]'],
         label='With P2P', marker='o', linestyle='-', linewidth=2)
plt.plot(grouped_data_without['Total Received Bytes'], grouped_data_without['Total Duration [s]'],
         label='Without P2P', marker='s', linestyle='-', linewidth=2)

plt.xlabel('File Size (Total Received Bytes)', fontsize=20)
plt.ylabel('Total Duration [s]', fontsize=20)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16) 

plt.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('total_duration_with_35_peers_with_and_without_p2p.png', format='png', dpi=300)
plt.show()

#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
# 13 - Total duration of the whole process with 50 peers and an increasing size of file with and without P2P

df['P2P Algorithm'] = df['P2P Algorithm'].astype(str)

df_with_super_peer = df[(df['P2P Algorithm']. str.upper() == 'USED') & (df['Number of Peers'] == 50)]

df_without_super_peer = df[(df['P2P Algorithm'].str.upper() == 'NOT USED') & (df['Number of Peers'] == 50)]


stats_with = df_with_super_peer.groupby('Total Received Bytes')['Total Duration [s]'].agg(['mean', 'std', 'count']).reset_index()
stats_without = df_without_super_peer.groupby('Total Received Bytes')['Total Duration [s]'].agg(['mean', 'std', 'count']).reset_index()

stats_with['sem'] = stats_with['std'] / np.sqrt(stats_with['count'])
stats_without['sem'] = stats_without['std'] / np.sqrt(stats_without['count'])

plt.figure(figsize=(12, 7), dpi=300)

plt.errorbar(stats_with['Total Received Bytes'], stats_with['mean'], yerr=stats_with['sem'],
             label='With P2P', marker='o', linestyle='-', linewidth=2, capsize=5)

plt.errorbar(stats_without['Total Received Bytes'], stats_without['mean'], yerr=stats_without['sem'],
             label='Without P2P', marker='s', linestyle='-', linewidth=2, capsize=5)

plt.xlabel('File Size (Total Received Bytes)', fontsize=20)
plt.ylabel('Total Duration [s]', fontsize=20)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('total_duration_with_50_peers_with_and_without_p2p_with_error_bars.png', format='png', dpi=300)
plt.show()
#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
# 14 - Total duration of the whole process with 75 peers and an increasing size of file with and without P2P

df['P2P Algorithm'] = df['P2P Algorithm'].astype(str)

df_with_super_peer = df[(df['P2P Algorithm'].str.upper() == 'USED') & (df['Number of Peers'] == 75)]

df_without_super_peer = df[(df['P2P Algorithm'].str.upper() == 'NOT USED') & (df['Number of Peers'] == 75)]

grouped_data_with = df_with_super_peer.groupby('Total Received Bytes')['Total Duration [s]'].mean().reset_index()
grouped_data_without = df_without_super_peer.groupby('Total Received Bytes')['Total Duration [s]'].mean().reset_index()

plt.figure(figsize=(12, 7), dpi=300)

plt.plot(grouped_data_with['Total Received Bytes'], grouped_data_with['Total Duration [s]'],
         label='With P2P', marker='o', linestyle='-', linewidth=2)
plt.plot(grouped_data_without['Total Received Bytes'], grouped_data_without['Total Duration [s]'],
         label='Without P2P', marker='s', linestyle='-', linewidth=2)

plt.xlabel('File Size (Total Received Bytes)', fontsize=20)
plt.ylabel('Total Duration [s]', fontsize=20)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16) 

plt.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('total_duration_with_75_peers_with_and_without_p2p.png', format='png', dpi=300)
plt.show()

#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
# Containerlab deploying and destroying with increasing number of peers

import pandas as pd
import matplotlib.pyplot as plt


data = {
    "Number of Peers": [5, 10, 20, 35, 50, 75],
    "Total Duration of Containerlab Deploying[s]": [37, 64, 118, 198, 279, 413],
    "Total Duration of Containerlab Destroying[s]": [1, 2, 3, 5, 6, 9]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Plotting the data
plt.figure(figsize=(12, 7), dpi=300)

plt.plot(df['Number of Peers'], df['Total Duration of Containerlab Deploying[s]'], label="Deploying of Containerlab [s]", marker='o', linestyle='-', color='blue', linewidth=2)
plt.plot(df['Number of Peers'], df['Total Duration of Containerlab Destroying[s]'],label="Destroying of Containerlab [s]", marker='o', linestyle='-', color='orange', linewidth=2)

plt.xlabel('Number of Peers')
plt.ylabel('Total Duration of Containerlab Deploying [s]')

plt.xticks(fontsize=16)
plt.yticks(fontsize=16) 

plt.legend()
plt.grid(True)
plt.show()

#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
# Define the provided data points for number of peers and the corresponding times for deploying and destroying Containerlab

import numpy as np

number_of_peers = np.array([5, 10, 20, 35, 50, 75])
deploying_times = np.array([37, 64, 118, 198, 279, 413])

slope, intercept = np.polyfit(number_of_peers, deploying_times, 1)

def calculate_time(number_of_peers, slope, intercept):
    return slope * number_of_peers + intercept

time_per_peer = calculate_time(1, slope, intercept)

slope, time_per_peer

#------------------------------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------------------------------
# CPU and Memory usage with increasing number of peers
data = {
    "Number of Peers": [5, 10, 20, 35, 50, 75],
    "Total Average CPU Usage (%)": [4.73, 7.30, 12.59, 16.12, 24.89, 37.61],
    "Total Average Memory Usage (%)": [5.61, 9.23, 15.66, 31.25, 34.35, 46.44]
}

df = pd.DataFrame(data)

plt.figure(figsize=(12, 7), dpi=300)

plt.plot(df["Number of Peers"], df["Total Average CPU Usage (%)"], label="CPU Usage (%)", marker='o', color='blue')
plt.plot(df["Number of Peers"], df["Total Average Memory Usage (%)"], label="Memory Usage (%)", marker='o', color='green')

plt.xlabel("Number of Peers")
plt.ylabel("Usage (%)")

plt.xticks(fontsize=16)
plt.yticks(fontsize=16) 

plt.legend()
plt.grid(True)
plt.show()


#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
# Avarage and Variance of upload and download speeds from real network data

import pandas as pd

file_path = '/Users/ozcankaraca/Desktop/Testresults/reduced-sample.csv'  

df = pd.read_csv(file_path)

mean_upload = df["Peak average maximum upload speed"].mean()
mean_download = df["Peak average maximum download speed"].mean()

variance_upload = df["Peak average maximum upload speed"].var()
variance_download = df["Peak average maximum download speed"].var()

print(f"Average upload speed: {mean_upload}")
print(f"Average download speed: {mean_download}")
print(f"Variance of upload speed: {variance_upload}")
print(f"Variance of download speed: {variance_download}")


#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
# Calculate of improvement for implemantation 2 with 20 peers

data20 = {
    "Test1": {"P2P": "USED", "Total Duration": 14.51, "Data Size": 2239815},
    "Test2": {"P2P": "USED", "Total Duration": 18.87, "Data Size": 4293938},
    "Test3": {"P2P": "USED", "Total Duration": 28.52, "Data Size": 8869498},
    "Test4": {"P2P": "USED", "Total Duration": 43.47, "Data Size": 15890720},
    "Test5": {"P2P": "USED", "Total Duration": 78.47, "Data Size": 32095088},
    "Test6": {"P2P": "USED", "Total Duration": 150.97, "Data Size": 67108864},
    "Test7": {"P2P": "USED", "Total Duration": 291.15, "Data Size": 134217728},
    "Test8": {"P2P": "NOT USED", "Total Duration": 14.27, "Data Size": 2239815},
    "Test9": {"P2P": "NOT USED", "Total Duration": 20.06, "Data Size": 4293938},
    "Test10": {"P2P": "NOT USED", "Total Duration": 33.79, "Data Size": 8869498},
    "Test11": {"P2P": "NOT USED", "Total Duration": 55.77, "Data Size": 15890720},
    "Test12": {"P2P": "NOT USED", "Total Duration": 104.550, "Data Size": 32095088},
    "Test13": {"P2P": "NOT USED", "Total Duration": 208.82, "Data Size": 67108864},
    "Test14": {"P2P": "NOT USED", "Total Duration": 408.19, "Data Size": 134217728}
}

percent_improvement = {}
for key, value in data20.items():
    if value["P2P"] == "USED":

        corresponding_test = next((k for k, v in data20.items() if v["P2P"] == "NOT USED" and v["Data Size"] == value["Data Size"]), None)
        if corresponding_test:

            duration_used = value["Total Duration"]
            duration_not_used = data20[corresponding_test]["Total Duration"]
            improvement = ((duration_not_used - duration_used) / duration_not_used) * 100
            percent_improvement[key] = improvement

percent_improvement


#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
# Calculate of improvement for implemantation 2 with 50 peers

data50 = {
    "Test1": {"P2P": "USED", "Total Duration": 27.26, "Data Size": 2239815},
    "Test2": {"P2P": "USED", "Total Duration": 34.79, "Data Size": 4293938},
    "Test3": {"P2P": "USED", "Total Duration": 54.92, "Data Size": 8869498},
    "Test4": {"P2P": "USED", "Total Duration": 82.48, "Data Size": 15890720},
    "Test5": {"P2P": "USED", "Total Duration": 160.35, "Data Size": 32095088},
    "Test6": {"P2P": "USED", "Total Duration": 319.80, "Data Size": 67108864},
    "Test7": {"P2P": "USED", "Total Duration": 635.45, "Data Size": 134217728},
    "Test8": {"P2P": "NOT USED", "Total Duration": 31.15, "Data Size": 2239815},
    "Test9": {"P2P": "NOT USED", "Total Duration": 45.54, "Data Size": 4293938},
    "Test10": {"P2P": "NOT USED", "Total Duration": 78.84, "Data Size": 8869498},
    "Test11": {"P2P": "NOT USED", "Total Duration": 129.49, "Data Size": 15890720},
    "Test12": {"P2P": "NOT USED", "Total Duration": 248.10, "Data Size": 32095088},
    "Test13": {"P2P": "NOT USED", "Total Duration": 501.19, "Data Size": 67108864},
    "Test14": {"P2P": "NOT USED", "Total Duration": 987.28, "Data Size": 134217728}
}

percent_improvement = {}
for key, value in data50.items():
    if value["P2P"] == "USED":

        corresponding_test = next((k for k, v in data50.items() if v["P2P"] == "NOT USED" and v["Data Size"] == value["Data Size"]), None)
        if corresponding_test:

            duration_used = value["Total Duration"]
            duration_not_used = data50[corresponding_test]["Total Duration"]
            improvement = ((duration_not_used - duration_used) / duration_not_used) * 100
            percent_improvement[key] = improvement

percent_improvement
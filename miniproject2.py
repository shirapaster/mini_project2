import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configure pandas to display the entire DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping

def calc_mean_erp(trial_points, ecog_data):
    """
    Calculate and plot the mean ERP for each finger movement.

    Args:
    trial_points (pd.DataFrame): DataFrame with columns ['start', 'peak', 'finger'].
    ecog_data (pd.DataFrame): DataFrame with a single column of brain signal.

    Returns:
    np.ndarray: 5x1201 matrix of averaged ERPs.
    """
    # Ensure column names and types are correct
    trial_points.columns = ['start', 'peak', 'finger']
    trial_points = trial_points.astype({'start': int, 'peak': int, 'finger': int})

    # Define constants for ERP window
    pre_event_duration = 200  # 200 ms before start
    post_event_duration = 1000  # 1000 ms after start
    total_samples = pre_event_duration + post_event_duration + 1  # Total 1201 samples

    # Initialize an empty matrix for ERPs
    erp_matrix = np.zeros((5, total_samples))

    # Process each finger movement
    for finger in range(1, 6):
        finger_trials = trial_points[trial_points['finger'] == finger]
        segments = []

        for _, trial in finger_trials.iterrows():
            start_idx = trial['start'] - pre_event_duration
            end_idx = trial['start'] + post_event_duration + 1

            # Only include segments within valid bounds
            if 0 <= start_idx < len(ecog_data) and end_idx <= len(ecog_data):
                segment = ecog_data.iloc[start_idx:end_idx, 0].to_numpy()
                segments.append(segment)

        # Compute the mean ERP for the finger
        if segments:
            erp_matrix[finger - 1, :] = np.mean(segments, axis=0)

    # Plot the ERPs
    time_axis = np.linspace(-200, 1000, total_samples)  # Time in ms
    plt.figure(figsize=(10, 6))
    for finger in range(5):
        plt.plot(time_axis, erp_matrix[finger], label=f'Finger {finger + 1}')
    plt.title('Mean ERP for Each Finger Movement')
    plt.xlabel('Time (ms)')
    plt.ylabel('Signal Amplitude')
    plt.legend()
    plt.grid()
    plt.show()

    return erp_matrix

# Load the data
print("Loading finger data...")
trial_points = pd.read_csv('events_file_ordered.csv', dtype={'start': int, 'peak': int, 'finger': int})
print("Finger data loaded successfully.")
print(trial_points.head())

print("Loading ECOG data...")
ecog_data = pd.read_csv('brain_data_channel_one.csv', header=None, names=['ECOG_Signal'])
print("ECOG data loaded successfully.")
print(ecog_data.head())

# Calculate the mean ERP
erp_result = calc_mean_erp(trial_points, ecog_data)

# Save the ERP matrix to a CSV file
fingers_erp_mean_df = pd.DataFrame(
    erp_result,
    index=[f'Finger {i+1}' for i in range(5)],
    columns=[f'Time {i}' for i in range(erp_result.shape[1])]
)
fingers_erp_mean_df.to_csv('fingers_erp_mean_output.csv', index=True)

# Display the ERP matrix as a DataFrame
print("\nThe 5x1201 matrix as a DataFrame:")
print(fingers_erp_mean_df)

print("ERP calculation completed and saved to 'fingers_erp_mean_output.csv'.")





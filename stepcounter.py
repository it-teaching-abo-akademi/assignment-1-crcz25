import argparse

import matplotlib.pyplot as plt
import numpy as np


# Simple function to visualize 4 arrays that are given to it
def visualize_data(timestamps, x_arr, y_arr, z_arr, s_arr):
    # Plotting accelerometer readings
    plt.figure(1)
    plt.plot(timestamps, x_arr, color="blue", linewidth=1.0)
    plt.plot(timestamps, y_arr, color="red", linewidth=1.0)
    plt.plot(timestamps, z_arr, color="green", linewidth=1.0)
    plt.show()
    # magnitude array calculation
    m_arr = []
    for i, x in enumerate(x_arr):
        m_arr.append(magnitude(x_arr[i], y_arr[i], z_arr[i]))
    plt.figure(2)
    # plotting magnitude and steps
    plt.plot(timestamps, s_arr, color="black", linewidth=1.0)
    plt.plot(timestamps, m_arr, color="red", linewidth=1.0)
    plt.show()


# Function to read the data from the log file
def read_data(filename):
    # Open the csv file for reading
    data = np.loadtxt(filename, delimiter=",", dtype=np.float64)
    # Reshape the data into separate arrays (time, x, y, z)
    timestamps = data[:, 0].astype(np.int64)
    x_array = data[:, 1]
    y_array = data[:, 2]
    z_array = data[:, 3]
    # Delete the data variable to free memory
    del data
    return timestamps, x_array, y_array, z_array


# Function to count steps.
# Should return an array of timestamps from when steps were detected
# Each value in this arrray should represent the time that step was made.
def count_steps(timestamps, x_arr, y_arr, z_arr):
    # See the pdf: pedometer-design-3-axis-digital-acceler fore more details

    # Declare parameters for step counting (using dynamic threshold and dynamic precision)
    window_size = 15  # Window size for the algorithm (adjustable)
    precision_threshold = 0.5  # Fixed precision threshold (adjustable)

    # Process the data based on the window size
    # Calculate the magnitude of the total acceleration
    magnitude_total_acc = []
    for x, y, z in zip(x_arr, y_arr, z_arr):
        magnitude_total_acc.append(magnitude(x, y, z))
    step_times = []  # Array to store the times when steps were detected
    sample_old = 0.0  # Initialize the previous sample
    sample_new = 0.0  # Initialize the current sample
    sample_result = 0.0  # Initialize the sample result
    thresholds = []  # For visualization
    maxs = []  # For visualization
    mins = []  # For visualization

    # Iterate over the data
    for i, time in enumerate(timestamps):
        # If the changes in acceleration are greater than a predefined precision, the newest sample result, is shifted to the sample_new register; otherwise the sample_new register will remain unchanged.
        sample_result = magnitude_total_acc[i]
        if abs(sample_result - sample_old) > precision_threshold:
            sample_new = sample_result
        else:
            sample_new = sample_old
        # The sample_old register is shifted to the sample_result register unconditionally
        sample_old = sample_result

        # Calculate the maximum and minimum acceleration in the window
        maximum = np.max(magnitude_total_acc[i : i + window_size])
        minimum = np.min(magnitude_total_acc[i : i + window_size])
        # Calculate the threshold for the window
        threshold = (maximum + minimum) / 2
        # A step is defined as happening if there is a negative slope of the acceleration plot (sample_new < sample_old) when the  acceleration curve crosses below the dynamic threshold.
        if sample_new < sample_old and sample_new < threshold:
            print("Step detected at time " + str(time))
            step_times.append(time)
        # Store the threshold for visualization
        thresholds.append(threshold)
        maxs.append(maximum)
        mins.append(minimum)

    # Plot the data
    fig, ax = plt.subplots()
    ax.plot(timestamps, magnitude_total_acc, color="blue", linewidth=1.0, label="data")
    ax.plot(timestamps, thresholds, color="orange", linewidth=1.0, label="threshold")
    ax.plot(timestamps, maxs, color="red", linewidth=1.0, label="max")
    ax.plot(timestamps, mins, color="black", linewidth=1.0, label="min")
    for step_time in step_times:
        ax.axvline(x=step_time, color="green", linewidth=1.0)
    ax.set(xlabel="time (ms)", ylabel="magnitude", title="Step counter")
    ax.legend()
    ax.grid()
    plt.show()

    # TODO: Actual implementation
    # rv = []
    # for i, time in enumerate(timestamps):
    #     if i == 0:
    #         rv.append(time)
    return step_times


def count_steps_advanced(timestamps, x_arr, y_arr, z_arr):
    return []


# Calculate the magnitude of the given vector
def magnitude(x, y, z):
    return np.linalg.norm((x, y, z))


# Function to convert array of times where steps happened into array to give into graph visualization
# Takes timestamp-array and array of times that step was detected as an input
# Returns an array where each entry is either zero if corresponding timestamp has no step detected or 50000 if the step was detected
def generate_step_array(timestamps, step_time):
    s_arr = []
    ctr = 0
    for i, time in enumerate(timestamps):
        if ctr < len(step_time) and step_time[ctr] <= time:
            ctr += 1
            s_arr.append(50000)
        else:
            s_arr.append(0)
    while len(s_arr) < len(timestamps):
        s_arr.append(0)
    return s_arr


# Check that the sizes of arrays match
def check_data(t, x, y, z):
    if len(t) != len(x) or len(y) != len(z) or len(x) != len(y):
        print("Arrays of incorrect length")
        return False
    print("The amount of data read from accelerometer is " + str(len(t)) + " entries")
    return True


def main():
    # Parse command line arguments to get the name of the file to read
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--filename", help="Name of the file to read the data from", type=str
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        help="Algorithm to use for step counting (simple or advanced)",
        type=str,
        default="simple",
    )
    args = parser.parse_args()
    file_name = args.filename
    algorithm = args.algorithm
    # Read data from a measurement file, change the inoput file name if needed
    timestamps, x_array, y_array, z_array = read_data("data/" + file_name)
    # Chek that the data does not produce errors
    if not check_data(timestamps, x_array, y_array, z_array):
        return
    # Check the type of algorithm to use for step counting
    if algorithm == "simple":
        print("Using simple algorithm for step counting")
        # Count the steps based on array of measurements from accelerometer
        st = count_steps(timestamps, x_array, y_array, z_array)
    else:
        print("Using advanced algorithm for step counting")
        st = count_steps_advanced(timestamps, x_array, y_array, z_array)
    # Print the result
    print(
        "This data contains " + str(len(st)) + " steps according to current algorithm"
    )
    # convert array of step times into graph-compatible format
    s_array = generate_step_array(timestamps, st)
    # visualize data and steps
    visualize_data(timestamps, x_array, y_array, z_array, s_array)


main()

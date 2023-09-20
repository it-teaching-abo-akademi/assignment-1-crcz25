import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

# Declare global variables
WINDOW_SIZE = 50  # Window size for the algorithm (adjustable)
PRECISION_THRESHOLD = 0.01  # Fixed precision threshold (adjustable)


# Simple function to visualize 4 arrays that are given to it
def visualize_data(timestamps, x_arr, y_arr, z_arr, s_arr):
    # Plotting accelerometer readings
    plt.figure(1)
    plt.plot(timestamps, x_arr, color="blue", linewidth=1.0, label="x")
    plt.plot(timestamps, y_arr, color="red", linewidth=1.0, label="y")
    plt.plot(timestamps, z_arr, color="green", linewidth=1.0, label="z")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    # magnitude array calculation
    m_arr = []
    for i, x in enumerate(x_arr):
        m_arr.append(magnitude(x_arr[i], y_arr[i], z_arr[i]))
    plt.figure(2)
    # plotting magnitude and steps
    plt.plot(timestamps, s_arr, color="black", linewidth=1.0)
    plt.plot(timestamps, m_arr, color="red", linewidth=1.0)
    plt.tight_layout()
    plt.savefig("detections.png")
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

    # Iterate over the data based on the window size
    # for i in range(len(magnitude_total_acc)):
    # print(i)
    for i in range(len(magnitude_total_acc) - WINDOW_SIZE):
        time = timestamps[i + WINDOW_SIZE]
        # Calculate the maximum and minimum acceleration in the window
        maximum = np.max(magnitude_total_acc[i : i + WINDOW_SIZE])
        minimum = np.min(magnitude_total_acc[i : i + WINDOW_SIZE])
        # Calculate the threshold for the window
        threshold = (maximum + minimum) / 2

        # If the changes in acceleration are greater than a predefined precision, the newest sample result, is shifted to the sample_new register; otherwise the sample_new register will remain unchanged.
        sample_result = magnitude_total_acc[i]
        if abs(sample_result - sample_new) > PRECISION_THRESHOLD:
            sample_new = sample_result
        else:
            sample_old = sample_new

        # A step is defined as happening if there is a negative slope of the acceleration plot (sample_new < sample_old) when the  acceleration curve crosses below the dynamic threshold.
        if sample_old >= threshold > sample_new and sample_new < sample_old:
            # print("Step detected at time " + str(time))
            step_times.append(time)
        # Store the threshold for visualization
        thresholds.append(threshold)
        maxs.append(maximum)
        mins.append(minimum)

    # Plot the data to compare the data and the thresholds
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    ax[0].plot(timestamps, magnitude_total_acc, color="blue", label="acc")
    ax[0].plot(
        timestamps[WINDOW_SIZE // 2 : len(timestamps) - WINDOW_SIZE // 2],
        thresholds,
        color="orange",
        label="threshold",
        linestyle="dashed",
    )
    ax[0].plot(
        timestamps[WINDOW_SIZE // 2 : len(timestamps) - WINDOW_SIZE // 2],
        maxs,
        color="red",
        label="max",
        linestyle="dashed",
    )
    ax[0].plot(
        timestamps[WINDOW_SIZE // 2 : len(timestamps) - WINDOW_SIZE // 2],
        mins,
        color="green",
        label="min",
        linestyle="dashed",
    )
    ax[0].set(
        xlabel="time (ms)", ylabel="magnitude", title="Step counter Thresholds (Full)"
    )
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(timestamps[:250], magnitude_total_acc[:250], color="blue", label="acc")
    ax[1].plot(
        timestamps[WINDOW_SIZE // 2 : 250 - WINDOW_SIZE // 2],
        thresholds[: 250 - WINDOW_SIZE],
        color="orange",
        label="threshold",
        linestyle="dashed",
    )
    ax[1].plot(
        timestamps[WINDOW_SIZE // 2 : 250 - WINDOW_SIZE // 2],
        maxs[: 250 - WINDOW_SIZE],
        color="red",
        label="max",
        linestyle="dashed",
    )
    ax[1].plot(
        timestamps[WINDOW_SIZE // 2 : 250 - WINDOW_SIZE // 2],
        mins[: 250 - WINDOW_SIZE],
        color="green",
        label="min",
        linestyle="dashed",
    )
    ax[1].set(
        xlabel="time (ms)",
        ylabel="magnitude",
        title="Step counter Thresholds (Zoomed in)",
    )
    ax[1].legend()
    ax[1].grid()
    plt.tight_layout()
    plt.savefig("stepcounter_simple_thresholds.png", bbox_inches="tight")
    plt.show()

    return step_times


def count_steps_advanced(timestamps, x_arr, y_arr, z_arr):
    # Create a gaussian kernel for filtering the data
    kernel = np.array([1, 4, 6, 4, 1]) / 16

    # Apply the gaussian filter to each dimension of the data to pre-smooth the signal
    x_arr_conv = np.convolve(x_arr, kernel, mode="same")
    y_arr_conv = np.convolve(y_arr, kernel, mode="same")
    z_arr_conv = np.convolve(z_arr, kernel, mode="same")

    # Plot the comparison between the original and filtered signals
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    ax[0].plot(timestamps, x_arr, color="blue", linewidth=1.5, label="x")
    ax[0].plot(timestamps, x_arr_conv, color="orange", linewidth=1.5, label="x conv")
    ax[0].set(xlabel="time (ms)", ylabel="magnitude", title="x acceleration")
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(timestamps, y_arr, color="blue", linewidth=1.5, label="y")
    ax[1].plot(timestamps, y_arr_conv, color="orange", linewidth=1.5, label="y conv")
    ax[1].set(xlabel="time (ms)", ylabel="magnitude", title="y acceleration")
    ax[1].legend()
    ax[1].grid()
    ax[2].plot(timestamps, z_arr, color="blue", linewidth=1.5, label="z")
    ax[2].plot(timestamps, z_arr_conv, color="orange", linewidth=1.5, label="z conv")
    ax[2].set(xlabel="time (ms)", ylabel="magnitude", title="z acceleration")
    ax[2].legend()
    ax[2].grid()
    plt.tight_layout()
    plt.savefig("stepcounter_adv_og_vs_3axis_filt.png")
    plt.show()

    # Filter the data using a low-pass filter (cutoff frequency 0.2 Hz)
    b, a = butter(4, 0.005, "low", analog=False)
    x_gravity = filtfilt(b, a, x_arr_conv)
    y_gravity = filtfilt(b, a, y_arr_conv)
    z_gravity = filtfilt(b, a, z_arr_conv)

    # Calculate the user acceleration by removing the gravity from the total acceleration
    x_user_acc = x_arr - x_gravity
    y_user_acc = y_arr - y_gravity
    z_user_acc = z_arr - z_gravity

    # Calculate the magnitude of the user acceleration
    magnitude_user_acc = [
        magnitude(x, y, z) for x, y, z in zip(x_user_acc, y_user_acc, z_user_acc)
    ]

    # Filter the data using a low-pass filter (cutoff frequency 5 Hz)
    b, a = butter(4, 0.1, "low", analog=False)
    x_user_acc = filtfilt(b, a, x_user_acc)
    y_user_acc = filtfilt(b, a, y_user_acc)
    z_user_acc = filtfilt(b, a, z_user_acc)

    # Filter the data using a high-pass filter (cutoff frequency 1 Hz)
    b, a = butter(4, 0.02, "high", analog=False)
    x_user_acc = filtfilt(b, a, x_user_acc)
    y_user_acc = filtfilt(b, a, y_user_acc)
    z_user_acc = filtfilt(b, a, z_user_acc)

    # Calculate the magnitude of the user acceleration
    magnitude_user_acc_filt = [
        magnitude(x, y, z) for x, y, z in zip(x_user_acc, y_user_acc, z_user_acc)
    ]

    # Calculate the steps using the acceleration of the user
    steps = count_steps(timestamps, x_user_acc, y_user_acc, z_user_acc)

    # Plot the data to compare the original and filtered signals
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(
        timestamps,
        magnitude_user_acc,
        color="blue",
        label="user acceleration",
    )
    ax[1].plot(
        timestamps,
        magnitude_user_acc_filt,
        color="orange",
        label="filtered user acceleration",
    )
    ax[0].set(xlabel="time (ms)", ylabel="magnitude", title="Step counter (advanced)")
    ax[1].set(xlabel="time (ms)", ylabel="magnitude", title="Step counter (advanced)")
    ax[0].legend()
    ax[0].grid()
    ax[1].legend()
    ax[1].grid()
    plt.tight_layout()
    plt.savefig("stepcounter_adv_og_vs_filt_user_acc.png")
    plt.show()

    return steps


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
            s_arr.append(100)
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
    parser.add_argument(
        "-w",
        "--window",
        help="Window size for the algorithm (adjustable)",
        type=int,
        default=50,
    )
    parser.add_argument(
        "-p",
        "--precision",
        help="Fixed precision threshold (adjustable)",
        type=float,
        default=0.2,
    )
    # Parse the arguments
    args = parser.parse_args()
    file_name = args.filename
    algorithm = args.algorithm
    global WINDOW_SIZE, PRECISION_THRESHOLD
    WINDOW_SIZE = args.window
    PRECISION_THRESHOLD = args.precision
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

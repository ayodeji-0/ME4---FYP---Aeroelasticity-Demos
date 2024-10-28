# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from io import BytesIO

# # Sidebar parameters
# st.sidebar.header("Flutter Parameters")
# initial_angle = st.sidebar.slider("Initial Angular Displacement (rad)", 0.1, 1.0, 0.5, 0.05)
# frequency = st.sidebar.slider("Frequency (Hz)", 1.0, 5.0, 2.5, 0.1)
# damping_factor = st.sidebar.slider("Damping Factor", 0.01, 0.1, 0.03, 0.01)
# duration = st.sidebar.slider("Duration (s)", 5, 20, 10, 1)

# # Generate time series
# fps = 30  # frames per second
# t = np.linspace(0, duration, duration * fps)

# # Flutter simulation: Increasing amplitude with sinusoidal oscillation
# def torsional_flutter(t, initial_angle, frequency, damping):
#     return initial_angle * np.exp(damping * t) * np.sin(2 * np.pi * frequency * t)

# angle = torsional_flutter(t, initial_angle, frequency, damping_factor)

# # Plot setup
# fig, ax = plt.subplots()
# ax.set_xlim(0, duration)
# ax.set_ylim(-1.5 * initial_angle, 1.5 * initial_angle)
# line, = ax.plot([], [], lw=2)

# # Animation function
# def animate(i):
#     line.set_data(t[:i], angle[:i])
#     return line,

# # Create animation
# ani = FuncAnimation(fig, animate, frames=len(t), interval=1000/fps, blit=True)

# # Display the animation in Streamlit
# st.write("### Torsional Flutter Simulation")
# st.write(f"Parameters - Initial Angle: {initial_angle} rad, Frequency: {frequency} Hz, Damping: {damping_factor}")
# st.pyplot(fig)


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from io import BytesIO

# Sidebar parameters
st.sidebar.header("Flutter Parameters")
initial_angle = st.sidebar.slider("Initial Angular Displacement (degrees)", 1, 30, 15, 1)
frequency = st.sidebar.slider("Frequency (Hz)", 0.1, 5.0, 1.0, 0.1)
damping_factor = st.sidebar.slider("Damping Factor", 0.01, 0.1, 0.03, 0.01)
duration = st.sidebar.slider("Duration (s)", 5, 20, 10, 1)

# Convert initial angle to radians
initial_angle_rad = np.radians(initial_angle)

# Generate time series
fps = 30  # frames per second
t = np.linspace(0, duration, duration * fps)

# Flutter simulation function
def torsional_flutter(t, initial_angle, frequency, damping):
    return initial_angle * np.exp(damping * t) * np.sin(2 * np.pi * frequency * t)

angle = torsional_flutter(t, initial_angle_rad, frequency, damping_factor)

# Plot setup
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# Define airfoil as a line that rotates around its center
line, = ax.plot([], [], 'b-', lw=3)

# Animation function
def animate(i):
    theta = angle[i]  # Current angle in radians
    # Define the airfoil's endpoints
    x = [0, np.cos(theta)]
    y = [0, np.sin(theta)]
    line.set_data(x, y)
    return line,

# Create animation
ani = FuncAnimation(fig, animate, frames=len(t), interval=1000/fps, blit=True)

# Display the animation in Streamlit
st.write("### Torsional Flutter Simulation of an Airfoil")
st.write(f"Parameters - Initial Angle: {initial_angle}°, Frequency: {frequency} Hz, Damping: {damping_factor}")
st.pyplot(fig)

# Export as GIF (optional)
gif_file = BytesIO()
ani.save(f"{gif_file}.gif", fps=fps)
st.write("Download the simulation as GIF:")
st.download_button("Download GIF", data=gif_file, file_name="flutter_simulation.gif", mime="image/gif")
st.image(gif_file.getvalue(), use_column_width=True)

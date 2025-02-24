import streamlit as st
from modules.Blisks import *

st.title("Blisk Flutter Analysis")

# User Inputs
num_blades = st.slider("Number of Blades", 10, 50, 20)
blade_length = st.slider("Blade Length (m)", 0.05, 0.2, 0.1)
disk_radius = st.slider("Disk Radius (m)", 0.3, 1.0, 0.5)

# Create and Display Blisk
blisk = Blisk(num_blades, blade_length, disk_radius)
st.pyplot(blisk.plot_blisk())

st.header("Flutter Analysis")
mass_ratio = st.slider("Mass Ratio", 0.1, 10.0, 1.0)
frequency_ratio = st.slider("Frequency Ratio", 0.1, 5.0, 1.0)
reduced_velocity = st.slider("Reduced Velocity", 0.1, 20.0, 5.0)

analysis = BliskFlutterAnalysis(mass_ratio, frequency_ratio, reduced_velocity)
eigenvalues, damping_ratios = analysis.get_results()

st.write("### Eigenvalues:")
st.write(eigenvalues)

st.write("### Damping Ratios:")
st.write(damping_ratios)

import streamlit as st
import plotly.graph_objects as go
from modules.CBT_Flutter import Airfoil, FlutterAnalysis

st.set_page_config(layout="wide", page_icon="./icon.ico")

## CSS Styling
st.markdown(
    """
    <style>
        .column-header { text-align: center; font-size: 20px; font-weight: bold; color: #333; background-color: #999999; padding: 10px; border-radius: 15px; }
        .column-header2 { text-align: center; font-size: 15px; font-weight: bold; padding: 10px; }
        .body { text-align: justify; font-size: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

## Session State Setup
if "airfoil_obj" not in st.session_state:
    st.session_state.airfoil_obj = None

if "show_preview" not in st.session_state:
    st.session_state.show_preview = True

if "prev_params" not in st.session_state:
    st.session_state.prev_params = None  # To track changes in sliders

## Page Setup
st.title('Coupled Bending Torsion Flutter')
col1, col2, col3 = st.columns([0.5, 1, 0.5])

## Parameters
with col1:
    st.markdown('<div class="column-header">Parameters</div>', unsafe_allow_html=True)
    col1_1, col1_2 = st.columns(2)

    with col1_1:
        st.markdown('<div class="column-header2">Design your Airfoil</div>', unsafe_allow_html=True)
        cont1_1 = st.container(border=True)
        with cont1_1:
            max_camber = st.slider('Max Camber', 0, 9, 0, 1)
            camber_position = st.slider('Camber Position', 0, 9, 0, 1)
            thickness = st.slider('Thickness', 0, 40, 12, 1)
            length = st.slider('Length', 0, 10, 1, 1)
            num_points = st.slider('Number of Discretization Points', 10, 100, 100, 1)
            centrepos = st.slider('Centre Position', 0.0, 1.0, 0.5, 0.01)
            show_preview = st.checkbox('Show Preview', value=st.session_state.show_preview)

        buton_11 = st.button('Generate Airfoil', use_container_width=True)

        # Store current slider values as a tuple
        current_params = (max_camber, camber_position, thickness, num_points, length, centrepos)

        # Check if any parameter has changed
        if st.session_state.prev_params is None or current_params != st.session_state.prev_params:
            st.session_state.airfoil_obj = None  # Reset airfoil
            st.session_state.prev_params = current_params  # Update stored parameters

        if buton_11:
            st.session_state.airfoil_obj = Airfoil(max_camber, camber_position, thickness, num_points, length, centrepos)
            st.session_state.airfoil_obj.generate_naca_airfoil4()

        # Persist the preview state
        st.session_state.show_preview = show_preview

        # Display the preview only if airfoil is not None and user wants to see it
        if st.session_state.airfoil_obj is not None and st.session_state.show_preview:
            st.markdown(f'<div class="column-header2">Airfoil {max_camber}{camber_position}{thickness} Preview</div>', unsafe_allow_html=True)

            # Convert Matplotlib plot to Plotly for interactivity
            coords = st.session_state.airfoil_obj.coords
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=coords[:, 0], y=coords[:, 1], mode='lines', line=dict(color='black')))
            fig.update_layout(title=f"Airfoil {st.session_state.airfoil_obj.code}", xaxis_title="X", yaxis_title="Y", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        button_12 = st.button('Start Analysis', use_container_width=True)

    with col1_2:
        st.markdown('<div class="column-header2">Define Flutter Parameters</div>', unsafe_allow_html=True)
        cont1_3 = st.container(border=True)
        with cont1_3:
            mu = st.slider('Mass Ratio', 0.0, 10.0, 0.0, 0.1)
            sigma = st.slider('Frequency Ratio', 0.0, 10.0, 0.0, 0.1)
            V = st.slider('Reduced Velocity', 0.0, 100.0, 0.0, 0.1)
            a = st.slider('Torsional Axis Location', 0.0, 1.0, 0.5, 0.01)
            b = st.slider('Semi-Chord Length', 0.0, 1.0, 0.5, 0.01)
            e = st.slider('Eccentricity', 0.0, 1.0, 0.5, 0.01)
            r = st.slider('Radius of Gyration', 0.0, 1.0, 0.5, 0.01)
            w_theta = st.slider('Torsional Vibrations Frequency', 0.0, 1000.0, 100.0, 0.1)
            mode = st.selectbox('Aerodynamic Influence Matrix', ['Steady - State Space', 'Quasi-Steady - State Space'])

    # TODO: Implement keep everything constant and vary one parameter functionality

## Results
with col2:
    st.markdown('<div class="column-header">Results</div>', unsafe_allow_html=True)

    if button_12:
        flutter_analysis = FlutterAnalysis(mu, sigma, V, a, b, e, r, mode, w_theta)
        flutter_analysis.compute_response()
        st.markdown('<div class="column-header2">Flutter Analysis</div>', unsafe_allow_html=True)

        st.write("### Eigenvalues:")
        st.write(flutter_analysis.vals)

        st.write("### Damping Ratios:")
        st.write(flutter_analysis.zeta)

        st.write("### Frequencies:")
        st.write(flutter_analysis.omega)

## Theory
with col3:
    st.markdown('<div class="column-header">Background</div>', unsafe_allow_html=True)

    st.markdown('<div class="column-header2">Introduction</div>', unsafe_allow_html=True)
    cont3_1 = st.container(border=True)
    with cont3_1:
        st.write('Add some text here')

    st.markdown('<div class="column-header2">Underlying Theory</div>', unsafe_allow_html=True)
    cont3_2 = st.container(border=True)
    with cont3_2:
        st.markdown('<div class="body">Add some text here</div>', unsafe_allow_html=True)

    st.markdown('<div class="column-header2">Problem Definition</div>', unsafe_allow_html=True)
    cont3_3 = st.container(border=True)
    with cont3_3:
        st.write('Add some text here')

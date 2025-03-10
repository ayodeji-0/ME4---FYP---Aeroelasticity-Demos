import streamlit as st
from modules.CBT_Flutter import *
from stylesheet import *


st.set_page_config(layout="wide", page_icon="./icon.ico")

max_width_()
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

if "sys_params" not in st.session_state:
    st.session_state.sys_params = None  # To track changes in system config

if "anim_properties" not in st.session_state:
    st.session_state.anim_properties = {
        "airfoil_color": "#ffffff",
        "transparency": 0.5,
    }


## Page Setup
st.title('Coupled Bending Torsion Flutter')
col1, col2, col3 = st.columns([0.5, 1, 0.5])

## Parameters
with col1:
    st.markdown('<div class="column-header">Parameters</div>', unsafe_allow_html=True)
    col1_1, col1_2, col1_3 = st.columns(3)

    with col1_1:
        st.markdown('<div class="column-header2">Design Airfoil</div>', unsafe_allow_html=True)
        cont1_1 = st.container(border=True)
        with cont1_1:
            max_camber = st.slider('Max Camber', 0, 9, 0, 1)
            camber_position = st.slider('Camber Position', 0, 9, 0, 1)
            thickness = st.slider('Thickness', 0, 40, 12, 1)
            length = st.slider('Length', 0, 10, 1, 1)
            num_points = st.slider('Discretization', 10, 100, 100, 1)
            centrepos = st.slider('Centre Position', 0.0, 1.0, 0.5, 0.01)
            show_preview = st.checkbox('Show Preview', value=st.session_state.show_preview)

        buton_11 = st.button('Generate Airfoil', use_container_width=True)

        # Store current slider values as a tuple
        current_sys_params = (max_camber, camber_position, thickness, num_points, length, centrepos)

        # Set properties to default if not present
        properties = st.session_state.anim_properties




        # Check if any parameter has changed
        if st.session_state.sys_params is None or current_sys_params != st.session_state.sys_params:
            st.session_state.airfoil_obj = None  # Reset airfoil
            st.session_state.sys_params = current_sys_params  # Update stored parameters
            

        if buton_11:
            st.session_state.airfoil_obj = Airfoil(max_camber, camber_position, thickness, num_points, length, centrepos)
            st.session_state.airfoil_obj.generate_naca_airfoil4()

        # Persist the preview state
        st.session_state.show_preview = show_preview

        # Display the preview only if airfoil is not None and user wants to see it
        if st.session_state.airfoil_obj is not None and st.session_state.show_preview:
            
            st.markdown(f'<div class="column-header2">Airfoil {max_camber}{camber_position}{thickness} Preview</div>', unsafe_allow_html=True)
            #st.write(f"Airfoil {max_camber}{camber_position}{thickness}")
            st.pyplot(st.session_state.airfoil_obj.plot(color=properties['airfoil_color']), use_container_width=True)#find plotly equivalent for interactive plot
        button_12 = st.button('Analyse', use_container_width=True)

    with col1_2:
        st.markdown('<div class="column-header2">Define System</div>', unsafe_allow_html=True)
        cont1_3 = st.container(border=True)
        with cont1_3:
            mu = st.slider('Mass Ratio · μ', 0.0, 20.0, 0.0, 0.1)
            sigma = st.slider('Frequency Ratio · σ', 0.0, 10.0, 0.0, 0.1)
            V = st.slider('Reduced Velocity · V', 0.0, 100.0, 0.0, 0.1)
            a = st.slider('Torsional Axis Location · a', 0.0, 1.0, 0.5, 0.01)
            b = st.slider('Semi-Chord Length · b', 0.0, 1.0, 0.5, 0.01)
            e = st.slider('Eccentricity · e', 0.0, 1.0, 0.5, 0.01)
            r = st.slider('Radius of Gyration · r', 0.0, 1.0, 0.5, 0.01)
            w_theta = st.slider('Torsional Vibrations Frequency · w\N{SUBSCRIPT TWO}', 0.0, 1000.0, 100.0, 0.1)
            mode = st.selectbox('Aerodynamic Influence', ['Steady - State Space', 'Quasi Steady - State Space'])

    with col1_3:
        st.markdown('<div class="column-header2">Animation Properties</div>', unsafe_allow_html=True)
        cont1_4 = st.container(border=True)
        with cont1_4:
            # Base animation properties - aesthetics
            properties['airfoil_color'] = st.color_picker('Airfoil Color', '#ffffff')
            properties['annotated_text_color'] = st.color_picker('Annotations Color', '#000000')
            properties['transparency'] = st.slider('Transparency', 0.0, 1.0, 0.5, 0.01)
            properties['show_chord'] = st.checkbox('Show Chord', value=True)
            properties['angled_text'] = st.checkbox('Show Angled Text', value=True)


            # Store the updated properties
            st.session_state.anim_properties = properties

            # Other animation properties - playback
            duration = st.slider('Duration · s', 1, 10, 10, 1)
            fps = st.slider('Frame Rate · fps', 0, 120, 30, 10)
            #st.write(f"Frame Count: {int(duration * fps)}")

        st.markdown('<div class="column-header2">Fix & Vary Parameters</div>', unsafe_allow_html=True)
        cont1_5 = st.container(border=True)
        with cont1_5:
            st.write('Add some text here')
            st.write('Add some checkboxes and sliders to fix one of n number of parameters and vary the rest')



    #Tdoo: Add keep everthing constant and vary x functionality, remove state space from mode names
    #: side by side colorpickers


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

## Column 2 Setup
cont2_width = 600
cont2_height = 600
## Results
with col2:
    st.markdown('<div class="column-header">Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="column-header2">Flutter Analysis</div>', unsafe_allow_html=True)

    with col2:


        # st.write("### Eigenvalues:")
        # st.write(fa.vals)

        # st.write("### Damping Ratios:")
        # st.write(fa.zeta)

        # st.write("### Frequencies:")
        # st.write(fa.omega)

        col2_1, col2_2 = st.columns(2)
        with col2_1:
            cont2_1 = st.container(height = cont2_height, border=True)
            
            with cont2_1:
                if button_12:

                    fa = FlutterAnalysis(mu, sigma, V, a, b, e, r, mode, w_theta)
                    fa.compute_response()
                    
                    
                    anim = fa.animate_flutter(st.session_state.airfoil_obj.coords, duration, fps, properties)

                    st.markdown('<div class="column-header2">Animations</div>', unsafe_allow_html=True)
                    #st.components.v1.html(anim.to_html5_video(), width=800, height=600, scrolling=False)
                    st.components.v1.html(anim, width = cont2_width,height =cont2_height, scrolling=True)#, width=800, height=600)

                    

            cont2_2 = st.container(border=True)
            with cont2_2:
                st.write('Add flutter plot here')
                #fa.
        with col2_2:
            cont2_3 = st.container(border=True)
            with cont2_3:
                st.write('Add amplitude and phase plot here')
                #fa.amp_phase_plot()
            cont2_4 = st.container(border=True)
            with cont2_4:
                st.write('Add frequency, damping ratio, and mode shape here')
                #fa.freq_damp_plot()
                st.write('Add stability plot here')
                #fa.stability_plot()
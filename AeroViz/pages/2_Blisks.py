import streamlit as st
from modules.Blisks import *

st.set_page_config(layout="wide", page_icon="./icons/icon.ico")

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

st.markdown(
    """
        <style>
                .stAppHeader {
                    background-color: rgba(255, 255, 255, 0.0);  /* Transparent background */
                    visibility: visible;  /* Ensure the header is visible */
                }

               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 1.5rem;
                    padding-right: 1.5rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)

## Session State Setup
if "blisk_obj" not in st.session_state:
    st.session_state.blisk_obj = None

if "show_preview" not in st.session_state:
    st.session_state.show_preview = True

if "sys_params" not in st.session_state:
    st.session_state.sys_params = None  # To track changes in system config

if "anim_properties" not in st.session_state:
    st.session_state.anim_properties = {
        "color": "#000000",
        "transparency": 0.5,
    }

## Page Setup
st.title('Structural Analysis of Bladed Disks')
col1, col2, col3 = st.columns([0.5, 1, 0.5])

## Parameters
with col1:
    st.markdown('<div class="column-header">Parameters</div>', unsafe_allow_html=True)
    col1_1, col1_2, col1_3 = st.columns(3)

    with col1_1:
        st.markdown('<div class="column-header2">Design Blisk</div>', unsafe_allow_html=True)
        cont1_1 = st.container(border=True)
        with cont1_1:
            mode = st.selectbox('Analysis Type', ['Dimensionless', 'Dimensional'])
            num_blades = st.slider('Number of Blades', 3, 20, 10, 1)
            num_points = st.slider('Discretization', 10, 500, 10, 10)
            radial_segments = st.slider('Radial Segments', 5, 20, 10, 1)
            blade_segments = st.slider('Blade Segments', 5, 20, 10, 1)
            blade_length = st.slider('Blade Length', 0.05, 0.2, 0.1, 0.01)
            disk_radius = st.slider('Disk Radius', 0.3, 1.0, 0.5, 0.01)

            if mode == 'Dimensional':
                blade_width = st.slider('Blade Width', 0.05, 0.2, 0.1, 0.01)
                thickness = st.slider('Thickness - Both (Into Page)', 0.01, 0.1, 0.05, 0.01)

                # Consider having material database with 3 to 4 materials with important properties for each to use in all analysis based on user selection
            elif mode == ' Dimensionless':
                dimensionless = st.slider('Dimensionless Parameter', 0.0, 1.0, 0.5, 0.01)
            show_preview = st.checkbox('Show Preview', value=st.session_state.show_preview)

        buton_11 = st.button('Generate Blisk', use_container_width=True)

        # Store current slider values as a tuple
        current_sys_params = (mode, num_blades, num_points, radial_segments, blade_segments, blade_length, disk_radius)

        # Set properties to default if not present
        properties = st.session_state.anim_properties




        # Check if any parameter has changed
        if st.session_state.sys_params is None or current_sys_params != st.session_state.sys_params:
            st.session_state.blisk_obj = None  # Reset airfoil
            st.session_state.sys_params = current_sys_params  # Update stored parameters
            

        if buton_11:
            st.session_state.blisk_obj = Blisk(num_blades=num_blades, blade_length=blade_length, disk_radius=disk_radius)
            #st.session_state.blisk_obj.generate_naca_airfoil4()

        # Persist the preview state
        st.session_state.show_preview = show_preview

        # Display the preview only if airfoil is not None and user wants to see it
        if st.session_state.blisk_obj is not None and st.session_state.show_preview:
            
            st.markdown(f'<div class="column-header2">{num_blades}-Blade Blisk Preview</div>', unsafe_allow_html=True)
            #st.write(f"Airfoil {max_camber}{camber_position}{thickness}")
            st.pyplot(st.session_state.blisk_obj.plot(color=properties['airfoil_color']), use_container_width=True)#find plotly equivalent for interactive plot
        button_12 = st.button('Analyse', use_container_width=True)

    with col1_2:
        st.markdown('<div class="column-header2">System Configuration</div>', unsafe_allow_html=True)
        cont1_3 = st.container(border=True)
        with cont1_3:
            # mu = st.slider('Mass Ratio · μ', 0.0, 20.0, 0.0, 0.1)
            # sigma = st.slider('Frequency Ratio · σ', 0.0, 10.0, 0.0, 0.1)
            # V = st.slider('Reduced Velocity · V', 0.0, 100.0, 0.0, 0.1)
            # a = st.slider('Torsional Axis Location · a', 0.0, 1.0, 0.5, 0.01)
            # b = st.slider('Semi-Chord Length · b', 0.0, 1.0, 0.5, 0.01)
            # e = st.slider('Eccentricity · e', 0.0, 1.0, 0.5, 0.01)
            # r = st.slider('Radius of Gyration · r', 0.0, 1.0, 0.5, 0.01)
            # w_theta = st.slider('Torsional Vibrations Frequency · w\N{SUBSCRIPT TWO}', 0.0, 1000.0, 100.0, 0.1)
            st.write("Replace sliders with blisk relevant parameters")

                # Consider having material database with 3 to 4 materials with important properties for each to use in all analysis based on user selection
                

    with col1_3:
        st.markdown('<div class="column-header2">Animation Properties</div>', unsafe_allow_html=True)
        cont1_4 = st.container(border=True)
        with cont1_4:
            # Base animation properties - aesthetics
            properties['airfoil_color'] = st.color_picker('Blisk Color', '#000000')
            properties['annotated_text_color'] = st.color_picker('Annotations Color', '#000000')
            properties['transparency'] = st.slider('Transparency', 0.0, 1.0, 0.5, 0.01)
            properties['show_nc'] = st.checkbox('Show Nodal Circles', value=True)
            properties['show_nds'] = st.checkbox('Show Nodal Diameters', value=True)


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

## Results
with col2:
    st.markdown('<div class="column-header">Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="column-header2">Structural Analysis</div>', unsafe_allow_html=True)
        # st.write("### Eigenvalues:")
        # st.write(fa.vals)

        # st.write("### Damping Ratios:")
        # st.write(fa.zeta)

        # st.write("### Frequencies:")
        # st.write(fa.omega)

    col2_1, col2_2 = st.columns(2)
    with col2_1:
        cont2_1 = st.container(border=True)
        
        with cont2_1:
            analysis = None
            if button_12:
                st.write("Blisk animation goes here")
                blisk_obj = Blisk(num_blades = num_blades, blade_length=blade_length, disk_radius=disk_radius, blade_segments=blade_segments, radial_segments=radial_segments)
                blisk_obj.precompute_parameters()
                st.pyplot(blisk_obj.plot())
                
                # Placeholder display
                analysis = BliskAnalysis(blisk_obj=blisk_obj, time=1, intervals=100)
                deformations = analysis.compute_deformations()
                st.write("### Deformations:")
                st.write(deformations)


                # fa = FlutterAnalysis(mu, sigma, V, a, b, e, r, mode, w_theta)
                # fa.compute_response()                
                # anim = fa.animate_flutter(st.session_state.blisk_obj.coords, duration, fps, properties)
                # st.markdown('<div class="column-header2">Animations</div>', unsafe_allow_html=True)
                # st.components.v1.html(anim, scrolling=True)#, width=800, height=600)

                

        cont2_2 = st.container(border=True)
        with cont2_2:
            st.write('Add mode shapes here')

    with col2_2:
        cont2_3 = st.container(border=True)
        with cont2_3:
            cont2_width = 800
            cont2_height = 600
            st.write('Add amplitude and phase plot here')
            if analysis is not None:
                st.write('Add blisk plot here')
                anim = analysis.animate_deformations()
                st.components.v1.html(anim, width = cont2_width,height =cont2_height, scrolling=True)#, width=800, height=600)

                #fa.
            #fa.amp_phase_plot()
        cont2_4 = st.container(border=True)
        with cont2_4:
            st.write('Add frequency, damping ratio, and mode shape here')
            #fa.freq_damp_plot()
            st.write('Add stability plot here')
            #fa.stability_plot()

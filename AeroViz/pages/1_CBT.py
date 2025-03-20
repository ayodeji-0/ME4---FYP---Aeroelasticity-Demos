import streamlit as st
from modules.CBT_Flutter import *
from stylesheet import *


st.set_page_config(layout="wide", page_icon="./icons/icon.ico")

max_width_()
## CSS Styling
# Text Styling
st.markdown(
    """
    <style>
        .column-header { text-align: center; font-size: 20px; font-weight: bold; color: #000000; background-color: #999999; padding: 10px; border-radius: 15px; }
        .column-header2 { text-align: center; font-size: 15px; font-weight: bold; padding: 10px; color:#ffffff; }
        .column-header3 { text-align: center; font-size: 15px; font-weight: bold; padding: 0px; background-color: #999999; border-radius: 15px; color: #000000; }
        .body { text-align: justify; font-size: 12px; color: #000000; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Tabs Styling
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;  /* Adjust spacing between tabs */
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 5px;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header Styling - page label vertical translation and background color
st.markdown(
    """
        <style>
                .stAppHeader {
                    background-color: rgba(255, 255, 255, 0.0);  /* Transparent background */
                    visibility: visible;  /* Ensure the header is visible */
                }

               .block-container {
                    background-color: rgba(255, 255, 255, 0.0);  /* Transparent background */
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 1.5rem;
                    padding-right: 1.5rem;
                }
                .stContainer {
                    background-color: transparent !important;  /* Transparent background */
                }
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

if "fa" not in st.session_state:
    st.session_state.fa = None

## Page Setup
st.title('Coupled Bending Torsion Flutter')
col1, col2, col3 = st.columns([0.5, 1, 0.5])
cont_height = 1000
## Parameters
with col1:
    st.markdown('<div class="column-header">Parameters</div>', unsafe_allow_html=True)
    st.markdown('<div class="column-header2">Configure Section Model</div>', unsafe_allow_html=True)
    cont1_3 = st.container(border=True, height=cont_height)
    with cont1_3:
        col1_tabs = st.tabs(["Design Airfoil","System Configuration", "Animation Properties", "Parametric Study", "Plotting"])
        
        with col1_tabs[1]:
            col1_1, col1_2 = st.columns(2)
            with col1_1:            
                mu = st.slider('Mass Ratio · $μ$', 0.1, 20.0, 0.1, 0.1)
                sigma = st.slider('Frequency Ratio · $σ$', 0.1, 10.0, 0.1, 0.1)
                V = st.slider('Reduced Velocity · $V$', 0.1, 100.0, 0.1, 0.1)
                a = st.slider('Torsional Axis Location · $a$', 0.0, 1.0, 0.5, 0.01)
                b = st.slider('Semi-Chord Length · $b$', 0.0, 1.0, 0.5, 0.01)
                e = st.slider('Eccentricity · $e$', 0.0, 1.0, 0.5, 0.01)
                r = st.slider('Radius of Gyration · $r$', 0.0, 1.0, 0.5, 0.01)
                w_theta = st.slider('Torsional Vibration Frequency · $w_{\\theta}$', 0.0, 1000.0, 100.0, 0.1)
                mode = st.selectbox('Aerodynamic Influence', ['Steady - State Space', 'Quasi Steady - State Space'])

                # Perform System Analysis
                button_12 = st.button('Run Analysis', use_container_width=True)
            with col1_2:
                cont1_1_1 = st.container(border=True)
                with cont1_1_1:
                    st.markdown('<div class="column-header2">Key Numerical Results</div>', unsafe_allow_html=True)
                    buff = st.empty()
                    with buff:
                        st.info('Results will be displayed here!', icon="📊")
                if button_12:
                    with cont1_1_1:
                        buff.empty()
                        # Define the flutter analysis object
                        st.session_state.fa = FlutterAnalysis(mu, sigma, V, a, b, e, r, mode, w_theta)
                        # Compute the flutter response
                        st.session_state.fa.compute_response()

                        
                        with buff:
                            st.info('Loading Numerical Results...')#, icon="./icons/calculator.ico")
                        
                        st.write("Eigenvalues:")
                        st.write(st.session_state.fa.vals)
                        st.write("Damping Ratios:")
                        st.write(st.session_state.fa.zeta)
                        st.write("Frequencies:")
                        st.write(st.session_state.fa.omega)
                        with buff:
                            st.info('Results Loaded!', icon="✅")
                        buff.empty()
        with col1_tabs[0]:
    
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
            
            
            

        with col1_tabs[2]:
            st.markdown('<div class="column-header3">Aesthetics</div>', unsafe_allow_html=True)
            # Base animation properties - aesthetics
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                properties['airfoil_color'] = st.color_picker('Airfoil Color', '#ffffff')
            with col1_2:
                properties['annotated_text_color'] = st.color_picker('Annotation Color', '#000000')
                
            properties['transparency'] = st.slider('Airfoil Transparency', 0.0, 1.0, 0.5, 0.01)
            properties['show_chord'] = st.checkbox('Show Chord', value=True)
            properties['angled_text'] = st.checkbox('Show Angled Text', value=True)


            # Store the updated properties
            st.session_state.anim_properties = properties
            st.markdown('<div class="column-header3">Playback</div>', unsafe_allow_html=True)
            # Other animation properties - playback
            duration = st.slider('Duration · s', 1, 10, 10, 1)
            fps = st.slider('Frame Rate · fps', 0, 120, 30, 10)
            #st.write(f"Frame Count: {int(duration * fps)}")

        with col1_tabs[3]:
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                #st.markdown('<div class="body">Perform a parametric study to investigate the effect of varying system parameters on flutter characteristics. Select the parameters to vary and fix, along with the range and step size for each parameter. The study will generate a plot showing the variation in flutter speed with the selected parameters.</div>', unsafe_allow_html=True)
                st.write('Perform a parametric study to investigate the effect of varying system parameters on motion characteristics. Select the parameters to vary and fix, along with the range and step size for each parameter. The study will generate a plot showing the variation in flutter speed with the selected parameters.')
                study_param_x = st.selectbox('Select Parameter to Vary', ['Mass Ratio · μ', 'Frequency Ratio · σ', 'Reduced Velocity · V', 'Torsional Axis Location · a', 'Semi-Chord Length · b', 'Eccentricity · e', 'Radius of Gyration · r', 'Torsional Vibration Frequency · w_{θ}'])
                #study_param_y = st.selectbox('')
                #study_param_y = st.multiselect('Select Parameter to Fix', ['Mass Ratio · $μ$', 'Frequency Ratio · $σ$', 'Reduced Velocity · $V$', 'Torsional Axis Location · $a$', 'Semi-Chord Length · $b$', 'Eccentricity · $e$', 'Radius of Gyration · $r$', 'Torsional Vibration Frequency · $w_{\\theta}$'])
                st.slider('Start Value', 0.1, 100.0, 0.1, 0.1)
                st.slider('End Value', 0.1, 100.0, 0.1, 0.1)
                st.slider('Step Size', 0.1, 100.0, 0.1, 0.1)
                #'st.multiselect('')
                param_button = st.button('Run Parametric Study', use_container_width=True)
            with col1_2:
                cont1_1_2 = st.container(border=True)
                with cont1_1_2:
                    st.markdown('<div class="column-header2">Parametric Study Results</div>', unsafe_allow_html=True)
                    buff = st.empty()
                    with buff:
                        st.info('Results will be displayed here!', icon="📊")
        with col1_tabs[4]:
            button_1_5 = st.button('Reset Plots', use_container_width=True)
            button_1_6 = st.button('Reset ZZZZ', use_container_width=True)   

            if button_1_5:
                st.session_state.airfoil_obj = None
    #Tdoo: Add keep everthing constant and vary x functionality, remove state space from mode names
    #: side by side colorpickers


## Column 2 Setup
cont2_width = 600

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
        cont2_1 = st.container(height = cont_height, border=True)
        with cont2_1: 
            col2_tabs = st.tabs(["Displacement Plot", "Parametric Study","Incoming Feature", "Animation", "Reset Plots"])
            #col2_1, col2_2 = st.columns(2)
            with col2_tabs[3]:
                
                
                    button_2_1 = st.button('Animate Displacements', use_container_width=True)
                    if button_2_1:
                        # Quick Test against model workbook
                        # mu = 0.1
                        # sigma = 0.1
                        # V = 1
                        # a = 0.5
                        # b = 0.5
                        # e = 0.25
                        # r = 0.1
                        # mode = 'Steady - State Space'

                        # fa = FlutterAnalysis(mu, sigma, V, a, b, e, r, mode, w_theta)
                        # fa.compute_response()
                        # st.components.v1.html(fa.animate_flutter(st.session_state.airfoil_obj.coords, duration, fps, properties), width = cont2_width,height =cont_height, scrolling=True)

                        # st.write(f"Damping Ratios: {fa.zeta}")
                        # st.write(f"Frequencies: {fa.omega}")

                        buff = st.empty()
                        with buff:
                            st.info('Animating...', icon="🔍")
                        if st.session_state.fa and st.session_state.airfoil_obj is not None:
                            # Create animation if analysis is complete and airfoil is generated
                            anim = st.session_state.fa.animate_flutter(st.session_state.airfoil_obj.coords, duration, fps, properties)
                            st.markdown('<div class="column-header2">Animation</div>', unsafe_allow_html=True)
                            #st.components.v1.html(anim.to_html5_video(), width=800, height=600, scrolling=False)
                            st.components.v1.html(anim, width = cont2_width,height =cont_height, scrolling=True)#, width=800, height=600)
                        else:
                            st.error('Generate an Arifoil then Run Analysis First!', icon="⚠️")
                        with buff:
                            st.info('Rendering Complete!', icon="✅")
                        buff.empty()
                        
                        

                        

            with col2_tabs[2]:
                button_2_3 = st.button('One more button', use_container_width=True)
        
            with col2_tabs[0]:
                button_2_2 = st.button('Plot Displacements Against Time', use_container_width=True)#
                if button_2_2:
                    buff = st.empty()
                    with buff:
                        st.info('Plotting...', icon="📊")
                    if st.session_state.fa is not None:
                        fig = st.session_state.fa.plot_displacements(duration=duration, width=cont2_width, height=cont_height)
                        st.markdown('<div class="column-header2">Displacement Plot</div>', unsafe_allow_html=True)
                    else:
                        st.error(' Go to System Configuration, then Run Analysis first!', icon="⚠️")
                    with buff:
                        st.info('Rendering Complete!', icon="✅")
                        
                    buff.empty()
                    st.pyplot(fig)
            with col2_tabs[1]:
                button_2_4 = st.button('Plot Parametric Study', use_container_width=True)
                if button_2_4:
                    st.write('Add parametric study plot here')


## Theory
with col3:
    st.markdown('<div class="column-header">Background</div>', unsafe_allow_html=True)

    st.markdown('<div class="column-header2">Theory</div>', unsafe_allow_html=True)
    
    cont3_1 = st.container(height=cont_height,border=True)
    with cont3_1:
        col3_tabs = st.tabs(["Introduction", "Aeroelastic Formulation", "System of Equations", "Eigenproblem Definition"])
    with col3_tabs[0]:
            #st.markdown('<div class ="column-header2">Aeroelastic Formulation</div>', unsafe_allow_html=True)
            st.write("""
    Flutter is a dynamic instability that arises from the interaction between aerodynamic, elastic, and inertial forces. 
    It belongs to the class of dynamic aeroelastic phenomena, which includes buffeting and gust response. Unlike these, flutter is self-excited, meaning it does not require an external force to sustain oscillations. 
    Instead, it occurs due to a feedback loop between structural deformation and unsteady aerodynamic loads. \n
    A widely used approach to study flutter is through the "Typical Section Model", which represents a cross-section of a wing or rotor blade. This model is simplified yet retains key flutter characteristics found in complex systems. It consists of an airfoil section elastically mounted on springs, with two degrees of freedom: \n
    1. Plunge motion $h$ – Up and down movement.
    2. Twist motion $θ$ – Rotation around a reference axis. \n
    The springs represent bending $k_{h}$ and torsional stiffness $k_{\\theta}$ of the structure. The aerodynamic forces act at the aerodynamic center $x_{AC}$, while the mass is centered at $x_{CM}$. The flutter instability arises when these two motions couple under aerodynamic forces, leading to exponential growth in oscillations.
                    """)

    with col3_tabs[2]:
        st.latex(r"""
                \begin{bmatrix}
                m & m b x_\theta \\
                m b x_\theta & I_\theta
                \end{bmatrix}
                \begin{bmatrix}
                \ddot{h} \\
                \ddot{\theta}
                \end{bmatrix}
                +
                \begin{bmatrix}
                k_h & 0 \\
                0   & k_\theta
                \end{bmatrix}
                \begin{bmatrix}
                h \\
                \theta
                \end{bmatrix}
                =
                \begin{bmatrix}
                F_h \\
                F_\theta
                \end{bmatrix}
                """)
        st.write("In Non-Dimensional Form:")
        st.latex(r"""
                incoming non-dimensional form
                    """)
        st.write(""" Where: \n
    - $m$: mass per unit span  
    - $I_{\\theta}$: mass moment of inertia  
    - $k_{h}$,  $k_{\\theta}$: bending and torsional stiffness  
    - $F_{h}$,  $F_{\\theta}$: aerodynamic forces in plunge and twist
                """)



        
#     st.markdown('<div class="column-header2">Problem Definition</div>', unsafe_allow_html=True)
#     cont3_3 = st.container(border=True)
#     with cont3_3:
#         st.markdown("The problem formulation in flutter analysis depends on the type of aerodynamic forces considered: \n")
#         if mode == 'Steady - State Space':
#             st.markdown(""" Considering steady aerodynamic forces, \n

# The aerodynamic forces are computed from instantaneous angles of attack.
# No contribution from plunge velocity or acceleration.
# The lift and moment coefficients remain constant and do not change with frequency.
# Equation of motion simplifies significantly, making it easier to solve for flutter speed.""")
#         if mode == 'Quasi Steady - State Space':
#             st.markdown(""" Considering quasi-steady aerodynamic forces, \n
                    
# The aerodynamic forces become functions of both the instantaneous angle of attack and its derivatives.
# Plunge velocity, angular acceleration, and pitch rate influence the aerodynamic loads.
# Requires solving a more complex coupled system, but provides a better approximation of real-world flutter behavior.
# """)
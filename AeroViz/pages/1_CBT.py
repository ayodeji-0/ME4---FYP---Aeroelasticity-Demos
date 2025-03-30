import streamlit as st
from modules.CBT_Flutter import *
from stylesheet import *


st.set_page_config(layout="wide", page_icon="./icons/icon.ico")

#max_width_()
## CSS Styling
# Text Styling
st.markdown(
    """
    <style>
        .column-header { text-align: center; font-size: 20px; font-weight: bold; color: #000000; background-color: #999999; padding: 10px; border-radius: 15px; }
        .column-header2 { text-align: center; font-size: 15px; font-weight: bold; padding: 0px; background-color: #999999; border-radius: 15px; color: #000000; }
        .column-header3 { text-align: center; font-size: 15px; font-weight: bold; padding: 10px; color:#ffffff; }
        .column-header4 { text-align: center; font-size: 15px; font-weight: bold; padding: 10px; color:#ffffff; text-decoration: underline; }
        
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



# # App Header Styling - page label vertical translation and background color
st.markdown(
    """
        <style>
                .stAppHeader {
                    background-color: rgba(255, 255, 255, 0.0);  /* Transparent background */
                    visibility: visible;  /* Ensure the header is visible */
                }

               .block-container {
                    padding-top: 0rem; !important;
                    padding-bottom: 0rem; !important;
                    padding-left: 1.5rem;
                    padding-right: 1.5rem;
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

if "airfoil_params" not in st.session_state:
    st.session_state.airfoil_params = None  # To track changes in airfoil params

if "sys_params" not in st.session_state:
    st.session_state.sys_params = None  # To track changes in system config

if "study_params" not in st.session_state:
    st.session_state.study_params = None  # To track changes in parametric study config

if "anim_properties" not in st.session_state:
    st.session_state.anim_properties = {
        "airfoil_color": "#ffffff",
        "transparency": 0.5,
    }

if "fa" not in st.session_state:
    st.session_state.fa = None # To store the flutter analysis object

if "ps" not in st.session_state:
    st.session_state.ps = None # To store the parametric study object

if "plots" not in st.session_state:
    st.session_state.plots = {
                                "Displacement":None,
                                "Parametric Study":None,
                                "Incoming Feature":None,
                                "Animation":None                              
                              } # To store plots avoiding clearing when changing tabs

## Page Setup
st.title('Coupled Bending Torsion Flutter')
col1, col2, col3 = st.columns([0.5, 1, 0.5])
cont_height = 1000
## Parameters
with col1:
    st.markdown('<div class="column-header">Parameters</div>', unsafe_allow_html=True)
    st.markdown('<div class="column-header3">Configure Section Model & Properties</div>', unsafe_allow_html=True)
    cont1_3 = st.container(border=True, height=cont_height)
    with cont1_3:
        col1_tabs = st.tabs(["Design Airfoil","System Configuration", "Aesthetics & Playback", "Parametric Study"])
        
        # Design Airfoil Tab
        with col1_tabs[0]:
    
            max_camber = st.slider('Max Camber', 0, 9, 0, 1)
            camber_position = st.slider('Camber Position', 0, 9, 0, 1)
            thickness = st.slider('Thickness', 0, 40, 12, 1)
            length = st.slider('Length', 0, 10, 1, 1)
            num_points = st.slider('Discretization', 10, 100, 100, 1)
            centrepos = st.slider('Centre Position', 0.0, 1.0, 0.5, 0.01)
            show_preview = st.checkbox('Show Preview', value=st.session_state.show_preview)

            buton_1_0_0 = st.button('Generate Airfoil', use_container_width=True)

            # Store current airfoil slider values as a tuple
            current_airfoil = (max_camber, camber_position, thickness, num_points, length, centrepos)

            # Set properties to default if not present
            properties = st.session_state.anim_properties

            # Check if any parameter has changed
            if st.session_state.airfoil_params is None or current_airfoil != st.session_state.airfoil_params:
                st.session_state.airfoil_obj = None  # Reset airfoil
                st.session_state.airfoil_params = current_airfoil  # Update stored parameters
                

            if buton_1_0_0:
                st.session_state.airfoil_obj = Airfoil(max_camber, camber_position, thickness, num_points, length, centrepos)
                st.session_state.airfoil_obj.generate_naca_airfoil4()

            # Persist the preview state
            st.session_state.show_preview = show_preview

            # Display the preview only if airfoil is not None and user wants to see it
            if st.session_state.airfoil_obj is not None and st.session_state.show_preview:
                st.markdown(f'<div class="column-header3">Airfoil {max_camber}{camber_position}{thickness} Preview</div>', unsafe_allow_html=True)
                #st.write(f"Airfoil {max_camber}{camber_position}{thickness}")
                st.pyplot(st.session_state.airfoil_obj.plot(color=properties['airfoil_color']), use_container_width=True)#find plotly equivalent for interactive plot

        # System Configuration Tab
        with col1_tabs[1]:
            col1_1, col1_2 = st.columns(2)
            with col1_1:            
                mu = st.slider('Mass Ratio · $μ$', 0.1, 20.0, 0.1, 0.1, help="Mass per unit span to stiffness ratio")
                sigma = st.slider('Frequency Ratio · $σ$', 0.1, 10.0, 0.1, 0.1)
                V = st.slider('Reduced Velocity · $V$', 0.1, 100.0, 0.1, 0.1)
                a = st.slider('Torsional Axis Location · $a$', 0.0, 1.0, 0.5, 0.01)
                b = st.slider('Semi-Chord Length · $b$', 0.0, 1.0, 0.5, 0.01)
                e = st.slider('Eccentricity · $e$', 0.0, 1.0, 0.5, 0.01)
                r = st.slider('Radius of Gyration · $r$', 0.0, 1.0, 0.5, 0.01)
                w_theta = st.slider('Torsional Vibration Frequency · $w_{\\theta}$', 0.0, 1000.0, 100.0, 0.1)
                mode = st.selectbox('Aerodynamic Influence', ['Steady - State Space', 'Quasi Steady - State Space'])

                
                # Store system parameters as a tuple
                sys_params = (mu, sigma, V, a, b, e, r, mode, w_theta)

                # Check if any parameter has changed
                if st.session_state.sys_params is None or sys_params != st.session_state.sys_params:
                    st.session_state.fa = None  # Reset analysis
                    st.session_state.sys_params = sys_params # Update stored parameters


            with col1_2:
                button_1_1_0 = st.button('Run Analysis', use_container_width=True)
                cont1_1_1 = st.container(border=True)
                with cont1_1_1:
                    # Perform System Analysis
                    st.markdown('<div class="column-header4">Key Numerical Results</div>', unsafe_allow_html=True)
                    buff = st.empty()
                    with buff:
                        st.info('Results will be displayed here!', icon="📊")
                if button_1_1_0:
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
                  
            
            

        with col1_tabs[2]:
            st.markdown('<div class="column-header2">Aesthetics</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="column-header2">Playback</div>', unsafe_allow_html=True)
            # Other animation properties - playback
            duration = st.slider('Duration · s', 1, 10, 10, 1)
            fps = st.slider('Frame Rate · fps', 0, 120, 30, 10)
            #st.write(f"Frame Count: {int(duration * fps)}")

        with col1_tabs[3]:
            
            st.write("""
                    Investigate the effects of varying system configuration parameters on response characteristics. \n
                    Select the parameter to vary, input the desired range and step size. \n
                    Select the dependent variable(s)* of interest.
                    """)

            col1_1, col1_2 = st.columns(2)
            with col1_1:
                #st.markdown('<div class="body">Perform a parametric study to investigate the effect of varying system parameters on flutter characteristics. Select the parameters to vary and fix, along with the range and step size for each parameter. The study will generate a plot showing the variation in flutter speed with the selected parameters.</div>', unsafe_allow_html=True)
                study_param_x = st.selectbox('Select Parameter to Vary', list(ps_indep_dict.keys()))
                study_param_y = st.multiselect('Select Dependent Variable(s)', list(ps_dep_dict.keys()))
                min_val = st.number_input('Start Value', 0.1, 1000.0, 0.1, 0.1)
                max_val = st.number_input('End Value', 0.1, 100.0, 0.1, 0.1)
                step = st.number_input('Step Size', 0.1, 100.0, 0.1, 0.1)

            with col1_2:
                
                button_1_3_0 = st.button('Run Study', use_container_width=True)
                cont1_1_2 = st.container(border=True)
                if button_1_3_0:
                    st.session_state.ps = ParametricStudy(study_param_x, min_val, max_val, step, study_param_y)
                    st.session_state.ps.run_study(sys_params=sys_params)
                    
                with cont1_1_2:
                    st.markdown('<div class="column-header4">Parametric Study Results</div>', unsafe_allow_html=True)
                    buff = st.empty()
                    with buff:
                        st.info('Results will be displayed here!', icon="📊")
                        st.write(study_param_y)
            
            st.write("""
                    *Note: To run the study you must first configure the system. \n
                    You can select multiple dependent variables to plot against the varying parameter.
                    """)
                        
    #Tdoo: Add keep everthing constant and vary x functionality, remove state space from mode names
    #: side by side colorpickers


## Column 2 Setup
cont2_width = 600

## Results
with col2:
    st.markdown('<div class="column-header">Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<div class="column-header3">System Response Analysis</div>', unsafe_allow_html=True)

    with col2:
        cont2_1 = st.container(height = cont_height, border=True)
        with cont2_1: 
            col2_tabs = st.tabs(["Displacement Plot", "Parametric Study","Incoming Feature", "Animation"])
            
            # Displacement Plot Tab
            with col2_tabs[0]:
                col2_0_1, col2_0_2 = st.columns([0.7,0.3])
                with col2_0_1:
                    button_2_0_0 = st.button('Plot Displacements Against Time', use_container_width=True)#
                with col2_0_2:
                    button_2_0_1 = st.button('Reset Displacement Plot', use_container_width=True)
                if button_2_0_0:
                    buff = st.empty()
                    with buff:
                        st.info('Plotting...', icon="📊")
                    if st.session_state.fa is not None:
                        fig_disp = st.session_state.fa.plot_displacements(duration=duration, width=cont2_width, height=cont_height)
                        st.session_state.plots["Displacement"] = fig_disp
                        st.markdown('<div class="column-header3">Displacement Plot</div>', unsafe_allow_html=True)
                    else:
                        st.error(' Go to System Configuration, then Run Analysis first!', icon="⚠️")
                    with buff:
                        st.info('Rendering Complete!', icon="✅")
                    
                    if st.session_state.plots["Displacement"] is not None:
                        buff.empty()
                        st.pyplot(st.session_state.plots["Displacement"], use_container_width=True)
                    else:
                        st.error('No data to plot. Run Analysis First!', icon="⚠️")
                if button_2_0_1:
                    st.session_state.plots["Displacement"] = None
            
            # Parametric Study Tab
            with col2_tabs[1]:
                col2_1_1, col2_1_2 = st.columns([0.7,0.3])
                with col2_1_1:
                    button_2_1_0 = st.button(f'Plot Dependent Variables Against {study_param_x}', use_container_width=True)#
                with col2_1_2:
                    button_2_1_1 = st.button('Reset Parametric Plot', use_container_width=True)
            
                if button_2_1_0:
                    buff = st.empty()
                    if st.session_state.ps is not None and st.session_state.ps.plot() is not None:
                        fig_ps = st.session_state.ps.plot()
                        st.session_state.plots["Parametric Study"] = fig_ps
                        st.markdown('<div class="column-header3">Parametric Study Plot</div>', unsafe_allow_html=True)
                     
                        st.pyplot(fig_ps)
                    else:
                        st.error('Results are not available. Run Parametric Study First!', icon="⚠️")

            # Incoming Feature Tab
            with col2_tabs[2]:
                col2_2_1, col2_2_2 = st.columns([0.7,0.3])
                with col2_2_1:
                    button_2_2_0 = st.button('Plot Incoming Feature', use_container_width=True)
                with col2_2_2:
                    button_2_2_1 = st.button('Reset Incoming Feature', use_container_width=True)
                if button_2_2_0:
                    st.write('Add incoming feature plot here')

                if button_2_2_1:
                    st.session_state.plots["Incoming Feature"] = None

                           
            # Animation Tab
            with col2_tabs[3]:
                    button_2_3_0 = st.button('Animate Displacements', use_container_width=True)
                    if button_2_3_0:
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
                            st.markdown('<div class="column-header3">Animation</div>', unsafe_allow_html=True)
                            #st.components.v1.html(anim.to_html5_video(), width=800, height=600, scrolling=False)
                            st.components.v1.html(anim, width = cont2_width,height =cont_height, scrolling=True)#, width=800, height=600)
                        else:
                            st.error('Generate an Arifoil then Run Analysis First!', icon="⚠️")
                        with buff:
                            st.info('Rendering Complete!', icon="✅")
                        buff.empty()


## Theory
with col3:
    st.markdown('<div class="column-header">Background</div>', unsafe_allow_html=True)

    st.markdown('<div class="column-header3">Theory</div>', unsafe_allow_html=True)
    
    cont3_1 = st.container(height=cont_height,border=True)
    with cont3_1:
        col3_tabs = st.tabs(["Introduction", "Aeroelastic Formulation", "System of Equations", "Eigenproblem Definition","Ask AI"])
    
    # Introduction Tab
    with col3_tabs[0]:
            #st.markdown('<div class ="column-header3">Aeroelastic Formulation</div>', unsafe_allow_html=True)
            st.write("""
                    Flutter is a dynamic instability that arises from the interaction between aerodynamic, elastic, and inertial forces. 
                     It belongs to the class of dynamic aeroelastic phenomena, which includes buffeting and gust response. Unlike these, flutter is self-excited, meaning it does not require an external force to sustain oscillations. 
                    Instead, it occurs due to a feedback loop between structural deformation and unsteady aerodynamic loads. \n
                    """)
            st.write("<Incoming image of problem>")
            st.write("""
                     A widely used approach to study flutter is through the "Typical Section Model", which represents a cross-section of a wing or rotor blade. This model is simplified yet retains key flutter characteristics found in complex systems. It consists of an airfoil section elastically mounted on springs, with two degrees of freedom: \n
                    1. Plunge motion $h$ – Up and down movement.
                    2. Twist motion $θ$ – Rotation around a reference axis. \n
                    The springs represent bending $k_{h}$ and torsional stiffness $k_{\\theta}$ of the structure. The aerodynamic forces act at the aerodynamic center $x_{AC}$, while the mass is centered at $x_{CM}$. The flutter instability arises when these two motions couple under aerodynamic forces, leading to exponential growth in oscillations.
                    """)

    # Aeroelastic Formulation Tab
    with col3_tabs[1]:
        st.markdown('<div class="column-header2">Aeroelastic Formulation</div>', unsafe_allow_html=True)

    # System of Equations Tab
    with col3_tabs[2]:
        st.markdown('<div class="column-header2">System of Equations</div>', unsafe_allow_html=True)
        st.markdown("The equations of motion for the typical section model are derived from the Euler-Bernoulli beam theory. The coupled bending-torsion equations are given by:")
        st.latex(r"""
                m \ddot{h} + m b x_\theta \ddot{\theta} + k_h h = F_h
                """)
        st.latex(r"""
                m b x_\theta \ddot{h} + I_\theta \ddot{\theta} + k_\theta \theta = F_\theta
                """)
        st.write("In Non-Dimensional Form:")
        st.latex(r"""
                \ddot{h} + \mu x_\theta \ddot{\theta} + h = \frac{F_h}{m k_h}
                """)
        st.latex(r"""
                \mu x_\theta \ddot{h} + \sigma \ddot{\theta} + \theta = \frac{F_\theta}{I_\theta}
                """)
    
    # Eigenproblem Definition Tab
    with col3_tabs[3]:
        # equations = st.latex(r"""
        #                     \begin{bmatrix}
        #                     m & m b x_\theta \\
        #                     m b x_\theta & I_\theta
        #                     \end{bmatrix}
        #                     \begin{bmatrix}
        #                     \ddot{h} \\
        #                     \ddot{\theta}
        #                     \end{bmatrix}
        #                     +
        #                     \begin{bmatrix}
        #                     k_h & 0 \\
        #                     0   & k_\theta
        #                     \end{bmatrix}
        #                     \begin{bmatrix}
        #                     h \\
        #                     \theta
        #                     \end{bmatrix}
        #                     =
        #                     \begin{bmatrix}
        #                     F_h \\
        #                     F_\theta
        #                     \end{bmatrix}
        #                     """),
        #                 st.write("In Non-Dimensional Form:")
        #                 st.latex(r"""
        #                         <incoming non-dimensional form>
        #                             """),
        #                 st.write("""
        #                         Where: \n
        #                         - $m$: mass per unit span  
        #                         - $I_{\\theta}$: mass moment of inertia  
        #                         - $k_{h}$,  $k_{\\theta}$: bending and torsional stiffness  
        #                         - $F_{h}$,  $F_{\\theta}$: aerodynamic forces in plunge and twist
        #                         """)

        st.markdown('<div class="column-header2">Problem Definition</div>', unsafe_allow_html=True)

        st.markdown("The problem formulation in flutter analysis depends on the type of aerodynamic forces considered: \n")
        if mode == 'Steady - State Space':
            st.markdown(""" 
            Considering steady aerodynamic forces, \n
        
            The aerodynamic forces are computed from instantaneous angles of attack.
            No contribution from plunge velocity or acceleration.
            The lift and moment coefficients remain constant and do not change with frequency.
            Equation of motion simplifies significantly, making it easier to solve for flutter speed.
            """)
        if mode == 'Quasi Steady - State Space':
            st.markdown(""" Considering quasi-steady aerodynamic forces, \n
                    
            The aerodynamic forces become functions of both the instantaneous angle of attack and its derivatives.
            Plunge velocity, angular acceleration, and pitch rate influence the aerodynamic loads.
            Requires solving a more complex coupled system, but provides a better approximation of real-world flutter behavior.
            """)
    
    # Ask AI Tab
    with col3_tabs[4]:
        st.chat_message('Need some clarification?')
        st.chat_input('Ask AeroViz AI a question')
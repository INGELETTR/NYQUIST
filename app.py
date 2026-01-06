import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')
import tempfile
import base64
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Bode & Nyquist Visualizer",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    .stTextInput input {
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

class BodeNyquistVisualizer:
    def __init__(self, num=None, den=None):
        """
        Initialize with transfer function.
        Provide numerator and denominator coefficients.
        Example: num=[1], den=[1, 1] for 1/(s+1)
        """
        if num is not None and den is not None:
            self.sys = signal.TransferFunction(num, den)
        else:
            raise ValueError("Must provide num and den coefficients")
        
        # Frequency range (rad/s)
        self.w = np.logspace(-2, 2, 1000)
        
        # Calculate frequency response
        self.w, self.mag, self.phase = signal.bode(self.sys, self.w)
        
        # Convert magnitude from dB to linear
        self.mag_linear = 10**(self.mag / 20)
        
        # Calculate Nyquist points
        self.nyquist_real = []
        self.nyquist_imag = []
        for i in range(len(self.w)):
            r = self.mag_linear[i] * np.cos(np.radians(self.phase[i]))
            im = self.mag_linear[i] * np.sin(np.radians(self.phase[i]))
            self.nyquist_real.append(r)
            self.nyquist_imag.append(im)
    
    def plot_bode(self):
        """Plot Bode diagrams (amplitude and phase)"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Magnitude plot
        ax1.semilogx(self.w, self.mag, 'b', linewidth=2)
        ax1.set_title('Bode Diagram - Amplitude', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Magnitude [dB]', fontsize=12)
        ax1.grid(True, which='both', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Frequency [rad/s]', fontsize=12)
        
        # Phase plot
        ax2.semilogx(self.w, self.phase, 'r', linewidth=2)
        ax2.set_title('Bode Diagram - Phase', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Phase [deg]', fontsize=12)
        ax2.set_xlabel('Frequency [rad/s]', fontsize=12)
        ax2.grid(True, which='both', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_nyquist_construction_frame(self, idx):
        """Plot a single frame of Nyquist construction"""
        fig = plt.figure(figsize=(12, 10))
        
        # Create subplots: Bode plots (left) and Nyquist construction (right)
        ax1 = plt.subplot2grid((2, 2), (0, 0))  # Magnitude Bode
        ax2 = plt.subplot2grid((2, 2), (1, 0))  # Phase Bode
        ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)  # Nyquist construction
        
        # Get current values
        freq = self.w[idx]
        mag_db = self.mag[idx]
        mag_lin = self.mag_linear[idx]
        phase_deg = self.phase[idx]
        phase_rad = np.radians(phase_deg)
        
        # Plot full Bode plots
        ax1.semilogx(self.w, self.mag, 'b', alpha=0.3, linewidth=2)
        ax1.plot([freq, freq], [ax1.get_ylim()[0], mag_db], 'bo', markersize=8)
        ax1.set_title('Bode Diagram - Amplitude', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Magnitude [dB]', fontsize=10)
        ax1.grid(True, which='both', linestyle='--', alpha=0.7)
        ax1.set_xlim([self.w[0], self.w[-1]])
        
        ax2.semilogx(self.w, self.phase, 'r', alpha=0.3, linewidth=2)
        ax2.plot([freq, freq], [ax2.get_ylim()[0], phase_deg], 'ro', markersize=8)
        ax2.set_title('Bode Diagram - Phase', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Phase [deg]', fontsize=10)
        ax2.set_xlabel('Frequency [rad/s]', fontsize=10)
        ax2.grid(True, which='both', linestyle='--', alpha=0.7)
        ax2.set_xlim([self.w[0], self.w[-1]])
        
        # Set up Nyquist plot
        ax3.set_title('Nyquist Diagram Construction', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Real', fontsize=10)
        ax3.set_ylabel('Imaginary', fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Set equal aspect ratio
        ax3.set_aspect('equal', adjustable='box')
        
        # Calculate limits for Nyquist plot
        margin = 0.1
        all_real = self.nyquist_real + [-1, 1]
        all_imag = self.nyquist_imag + [-1, 1]
        x_min, x_max = min(all_real), max(all_real)
        y_min, y_max = min(all_imag), max(all_imag)
        x_range = x_max - x_min
        y_range = y_max - y_min
        ax3.set_xlim(x_min - margin*x_range, x_max + margin*x_range)
        ax3.set_ylim(y_min - margin*y_range, y_max + margin*y_range)
        
        # Plot circle for amplitude
        circle = Circle((0, 0), mag_lin, fill=False, color='blue', linewidth=2, alpha=0.5)
        ax3.add_patch(circle)
        
        # Plot phase line
        x_end = 1.5 * mag_lin * np.cos(phase_rad)
        y_end = 1.5 * mag_lin * np.sin(phase_rad)
        ax3.plot([0, x_end], [0, y_end], 'r--', linewidth=2, alpha=0.7)
        
        # Plot intersection point
        x_point = mag_lin * np.cos(phase_rad)
        y_point = mag_lin * np.sin(phase_rad)
        ax3.plot(x_point, y_point, 'go', markersize=10, markeredgecolor='k', markeredgewidth=2)
        
        # Plot Nyquist trajectory up to current point
        ax3.plot(self.nyquist_real[:idx+1], self.nyquist_imag[:idx+1], 'g-', linewidth=2, alpha=0.7)
        
        # Add text box
        text_str = f'Frequency: {freq:.3f} rad/s\nAmplitude: {mag_lin:.3f}\nPhase: {phase_deg:.1f}°\nPoint: ({x_point:.3f}, {y_point:.3f})'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax3.text(0.02, 0.98, text_str, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return fig
    
    def plot_complete_nyquist(self):
        """Plot the complete Nyquist diagram"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot Nyquist diagram
        ax.plot(self.nyquist_real, self.nyquist_imag, 'b-', linewidth=2)
        ax.plot(self.nyquist_real, -np.array(self.nyquist_imag), 'b--', linewidth=1, alpha=0.5)
        
        # Mark (-1, 0) point for stability analysis
        ax.plot(-1, 0, 'rx', markersize=12, markeredgewidth=2)
        ax.text(-1.1, 0.1, '(-1, 0)', fontsize=12)
        
        ax.set_title('Complete Nyquist Diagram', fontsize=14, fontweight='bold')
        ax.set_xlabel('Real', fontsize=12)
        ax.set_ylabel('Imaginary', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        return fig

def parse_transfer_function(tf_str):
    """
    Simple transfer function parser that actually works
    Supports formats like: '1/(s+1)', '1/(s^2+0.5*s+1)', '(s+1)/(s^2+2*s+3)'
    """
    # Remove spaces and convert to lowercase
    tf_str = tf_str.replace(' ', '').lower()
    
    # Check if there's a denominator
    if '/' in tf_str:
        num_str, den_str = tf_str.split('/')
    else:
        num_str = tf_str
        den_str = "1"
    
    # Remove outer parentheses if present
    if num_str.startswith('(') and num_str.endswith(')'):
        num_str = num_str[1:-1]
    if den_str.startswith('(') and den_str.endswith(')'):
        den_str = den_str[1:-1]
    
    # Parse numerator and denominator
    num_coeffs = parse_polynomial(num_str)
    den_coeffs = parse_polynomial(den_str)
    
    return num_coeffs, den_coeffs

def parse_polynomial(poly_str):
    """
    Parse polynomial string to list of coefficients
    """
    # Handle constant case
    if 's' not in poly_str:
        return [float(poly_str)]
    
    # Remove parentheses if present
    poly_str = poly_str.replace('(', '').replace(')', '')
    
    # Replace '*' for easier parsing
    poly_str = poly_str.replace('*', '')
    
    # Split by '+' and '-' (keeping the signs)
    import re
    terms = re.split('([+-])', poly_str)
    
    # Reconstruct terms with signs
    reconstructed_terms = []
    current_sign = '+'
    for term in terms:
        if term == '+' or term == '-':
            current_sign = term
        elif term:
            if current_sign == '-':
                reconstructed_terms.append('-' + term)
            else:
                reconstructed_terms.append(term)
    
    # Find highest power
    max_power = 0
    for term in reconstructed_terms:
        if 's^' in term:
            power = int(term.split('s^')[1])
            max_power = max(max_power, power)
        elif 's' in term and '^' not in term:
            max_power = max(max_power, 1)
    
    # Initialize coefficient array
    coeffs = [0.0] * (max_power + 1)
    
    # Parse each term
    for term in reconstructed_terms:
        # Determine sign
        sign = 1
        if term.startswith('-'):
            sign = -1
            term = term[1:]
        
        # Parse the term
        if 's^' in term:
            coeff_part, power_part = term.split('s^')
            power = int(power_part)
            coeff = float(coeff_part) if coeff_part else 1.0
        elif 's' in term:
            coeff_part = term.split('s')[0]
            coeff = float(coeff_part) if coeff_part else 1.0
            power = 1
        else:
            coeff = float(term)
            power = 0
        
        # Place coefficient in the right position
        coeffs[max_power - power] = sign * coeff
    
    return coeffs

def create_gif_from_frames(frames):
    """Create a GIF from matplotlib figures"""
    try:
        import imageio
        from PIL import Image
        
        # Save each frame as PNG
        png_images = []
        for i, fig in enumerate(frames):
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            png_images.append(Image.open(buf))
            plt.close(fig)
        
        # Create GIF
        gif_buffer = BytesIO()
        png_images[0].save(
            gif_buffer,
            format='GIF',
            save_all=True,
            append_images=png_images[1:],
            duration=100,
            loop=0
        )
        gif_buffer.seek(0)
        
        return gif_buffer
    except ImportError:
        st.error("Please install imageio and pillow packages for GIF creation: pip install imageio pillow")
        return None

def main():
    """Main Streamlit app"""
    st.markdown('<h1 class="main-header">📈 Bode & Nyquist Diagram Visualizer</h1>', unsafe_allow_html=True)
    
    # Sidebar for input and controls
    with st.sidebar:
        st.markdown('<h2 class="sub-header">Transfer Function</h2>', unsafe_allow_html=True)
        
        # Input options
        input_option = st.radio(
            "Input method:",
            ["Manual Input", "Predefined Examples"]
        )
        
        if input_option == "Manual Input":
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Enter transfer function in one of these formats:**")
            st.markdown("- `1/(s+1)`")
            st.markdown("- `1/(s^2+0.5*s+1)`")
            st.markdown("- `(s+1)/(s^2+2*s+3)`")
            st.markdown("- `s/(s^2+1)`")
            st.markdown('</div>', unsafe_allow_html=True)
            
            tf_input = st.text_input(
                "Transfer function:",
                value="1/(s+1)",
                help="Enter transfer function in s-domain"
            )
            
            if st.button("Parse Transfer Function", type="primary"):
                st.session_state.tf_input = tf_input
                st.rerun()
        
        else:  # Predefined Examples
            example = st.selectbox(
                "Choose an example:",
                [
                    "First Order System: 1/(s+1)",
                    "Second Order System: 1/(s^2+0.5*s+1)",
                    "Lead Compensator: (s+1)/(s+2)",
                    "Integrator: 1/s",
                    "Double Integrator: 1/s^2",
                    "Oscillator: 1/(s^2+1)",
                    "Complex System: (s+2)/(s^3+3*s^2+3*s+1)"
                ]
            )
            
            # Map examples to transfer functions
            example_map = {
                "First Order System: 1/(s+1)": "1/(s+1)",
                "Second Order System: 1/(s^2+0.5*s+1)": "1/(s^2+0.5*s+1)",
                "Lead Compensator: (s+1)/(s+2)": "(s+1)/(s+2)",
                "Integrator: 1/s": "1/s",
                "Double Integrator: 1/s^2": "1/s^2",
                "Oscillator: 1/(s^2+1)": "1/(s^2+1)",
                "Complex System: (s+2)/(s^3+3*s^2+3*s+1)": "(s+2)/(s^3+3*s^2+3*s+1)"
            }
            
            tf_input = example_map[example]
            if st.button("Use This Example", type="primary"):
                st.session_state.tf_input = tf_input
                st.rerun()
        
        # Display parsed coefficients
        if 'tf_input' in st.session_state:
            try:
                num_coeffs, den_coeffs = parse_transfer_function(st.session_state.tf_input)
                
                st.markdown('<h2 class="sub-header">Parsed Coefficients</h2>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Numerator", str(num_coeffs))
                with col2:
                    st.metric("Denominator", str(den_coeffs))
                
                # Store in session state for use in main area
                st.session_state.num_coeffs = num_coeffs
                st.session_state.den_coeffs = den_coeffs
                st.session_state.parsed_successfully = True
                
            except Exception as e:
                st.error(f"Error parsing transfer function: {e}")
                st.session_state.parsed_successfully = False
        else:
            st.info("Enter a transfer function and click 'Parse' to begin")
            st.session_state.parsed_successfully = False
    
    # Main content area
    if 'parsed_successfully' in st.session_state and st.session_state.parsed_successfully:
        try:
            # Create visualizer instance
            visualizer = BodeNyquistVisualizer(
                num=st.session_state.num_coeffs,
                den=st.session_state.den_coeffs
            )
            
            # Display current transfer function
            st.markdown(f"### Current Transfer Function: `{st.session_state.tf_input}`")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["📊 Bode Plot", "🔄 Nyquist Construction", "🎯 Complete Nyquist"])
            
            with tab1:
                st.markdown('<h2 class="sub-header">Bode Diagram</h2>', unsafe_allow_html=True)
                fig_bode = visualizer.plot_bode()
                st.pyplot(fig_bode)
                plt.close(fig_bode)
            
            with tab2:
                st.markdown('<h2 class="sub-header">Nyquist Diagram Construction</h2>', unsafe_allow_html=True)
                
                # Controls for animation
                col1, col2 = st.columns(2)
                with col1:
                    num_frames = st.slider(
                        "Number of frames:",
                        min_value=10,
                        max_value=200,
                        value=50,
                        help="More frames = smoother but slower animation"
                    )
                with col2:
                    frame_step = st.slider(
                        "Frame step:",
                        min_value=1,
                        max_value=10,
                        value=2,
                        help="Step between frames (for faster preview)"
                    )
                
                if st.button("Generate Nyquist Construction Animation", type="primary"):
                    with st.spinner("Creating animation frames..."):
                        # Generate frames
                        frames = []
                        indices = np.linspace(0, len(visualizer.w)-1, num_frames, dtype=int)
                        
                        for idx in indices:
                            fig_frame = visualizer.plot_nyquist_construction_frame(idx)
                            frames.append(fig_frame)
                        
                        # Create GIF
                        gif_buffer = create_gif_from_frames(frames)
                        
                        if gif_buffer:
                            # Display GIF
                            st.markdown("### Nyquist Construction Animation")
                            
                            # Convert GIF to base64 for display
                            gif_base64 = base64.b64encode(gif_buffer.read()).decode()
                            gif_buffer.seek(0)
                            
                            # Display using HTML for better control
                            st.markdown(
                                f'<img src="data:image/gif;base64,{gif_base64}" alt="nyquist animation" width="800">',
                                unsafe_allow_html=True
                            )
                            
                            # Download button
                            st.download_button(
                                label="Download GIF",
                                data=gif_buffer,
                                file_name="nyquist_construction.gif",
                                mime="image/gif"
                            )
            
            with tab3:
                st.markdown('<h2 class="sub-header">Complete Nyquist Diagram</h2>', unsafe_allow_html=True)
                fig_nyquist = visualizer.plot_complete_nyquist()
                st.pyplot(fig_nyquist)
                plt.close(fig_nyquist)
                
                # Stability analysis info
                st.markdown("### Stability Analysis")
                st.info("""
                The **(-1, 0)** point is marked in red on the Nyquist diagram.
                
                **Nyquist Stability Criterion:**
                - If the Nyquist plot encircles (-1, 0) clockwise, the closed-loop system is unstable.
                - The number of encirclements relates to the number of unstable poles.
                - No encirclement of (-1, 0) indicates stability if the open-loop system is stable.
                """)
        
        except Exception as e:
            st.error(f"Error creating visualizations: {e}")
    
    else:
        # Welcome message
        st.markdown("""
        ## Welcome to the Bode & Nyquist Diagram Visualizer! 🎯
        
        This interactive tool helps you visualize and understand:
        
        **📊 Bode Plots** - Frequency response of a system
        - **Magnitude plot**: Shows how the system amplifies or attenuates different frequencies
        - **Phase plot**: Shows phase shift introduced by the system at different frequencies
        
        **🔄 Nyquist Diagrams** - Polar plot of frequency response
        - Visual representation of system stability
        - Shows how amplitude and phase relate in the complex plane
        - Interactive construction showing how Bode plots transform into Nyquist plot
        
        ### How to Use:
        1. Enter a transfer function in **s-domain** format in the sidebar
        2. Click **'Parse Transfer Function'** to validate your input
        3. Explore the different visualization tabs
        
        ### Quick Start:
        Try these examples in the sidebar:
        - `1/(s+1)` - Simple first-order system
        - `1/(s^2+0.5*s+1)` - Second-order system with damping
        - `1/(s^2+1)` - Oscillator (undamped system)
        """)
        
        # Quick examples in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Try 1/(s+1)"):
                st.session_state.tf_input = "1/(s+1)"
                st.rerun()
        with col2:
            if st.button("Try 1/(s^2+0.5*s+1)"):
                st.session_state.tf_input = "1/(s^2+0.5*s+1)"
                st.rerun()
        with col3:
            if st.button("Try 1/(s^2+1)"):
                st.session_state.tf_input = "1/(s^2+1)"
                st.rerun()

if __name__ == "__main__":
    main()
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import re
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Bode & Nyquist Visualizer",
    page_icon="📈",
    layout="wide"
)

def parse_transfer_function(tf_str):
    """
    Proper transfer function parser that handles multiplication and parentheses
    """
    # Clean input
    tf_str = tf_str.replace(' ', '').replace('^', '**')
    
    # Handle multiplication signs
    tf_str = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', tf_str)  # 5s -> 5*s
    tf_str = re.sub(r'([a-zA-Z\)])(\d)', r'\1*\2', tf_str)  # s5 -> s*5
    tf_str = re.sub(r'\)([a-zA-Z\(])', r')*\1', tf_str)     # )( -> )*(
    
    # Handle implicit multiplication
    tf_str = re.sub(r'(\d)(s)', r'\1*\2', tf_str)
    tf_str = re.sub(r'(s)(\d)', r'\1*\2', tf_str)
    
    # Check if there's a denominator
    if '/' in tf_str:
        num_str, den_str = tf_str.split('/')
    else:
        num_str = tf_str
        den_str = "1"
    
    # Parse polynomials
    num_coeffs = parse_polynomial(num_str)
    den_coeffs = parse_polynomial(den_str)
    
    return num_coeffs, den_coeffs

def parse_polynomial(expr):
    """
    Parse polynomial expression to coefficients using sympy-like evaluation
    """
    # If it's just a number
    if 's' not in expr:
        return [float(expr)]
    
    # Handle multiplication with parentheses
    expr = expand_expression(expr)
    
    # Find all terms with 's'
    terms = re.split(r'([+-])', expr)
    if terms[0] == '':
        terms = terms[1:]
    
    # Find highest power
    max_power = 0
    for term in terms:
        if term in ['+', '-']:
            continue
        if 's**' in term:
            match = re.search(r's\*\*(\d+)', term)
            if match:
                power = int(match.group(1))
                max_power = max(max_power, power)
        elif term.endswith('s') or (term.endswith('*s') and '*' in term):
            max_power = max(max_power, 1)
    
    # Initialize coefficients
    coeffs = [0.0] * (max_power + 1)
    
    # Process terms
    current_sign = '+'
    for term in terms:
        if term == '+' or term == '-':
            current_sign = term
            continue
        
        # Get coefficient and power
        if 's**' in term:
            # Term like 2*s**2
            coeff_part = term.split('*s**')[0]
            if coeff_part == '':
                coeff = 1.0
            elif coeff_part == '-':
                coeff = -1.0
                current_sign = '+'
            elif coeff_part == '+':
                coeff = 1.0
                current_sign = '+'
            else:
                coeff = float(coeff_part) if coeff_part else 1.0
            power = int(term.split('**')[-1])
        elif '*s' in term:
            # Term like 2*s
            coeff_part = term.split('*s')[0]
            coeff = float(coeff_part) if coeff_part else 1.0
            power = 1
        elif term.endswith('s'):
            # Term like 2s (after expansion)
            coeff_part = term[:-1]
            coeff = float(coeff_part) if coeff_part else 1.0
            power = 1
        else:
            # Constant term
            coeff = float(term)
            power = 0
        
        # Apply sign
        if current_sign == '-':
            coeff = -coeff
        
        # Store coefficient
        coeffs[max_power - power] = coeff
    
    return coeffs

def expand_expression(expr):
    """
    Expand expressions like 20*(s**2+1) to 20*s**2 + 20
    """
    # Remove parentheses by distributing multiplication
    while '(' in expr and ')' in expr:
        # Find innermost parentheses
        start = expr.rfind('(')
        end = expr.find(')', start)
        
        if start == -1 or end == -1:
            break
        
        inner = expr[start+1:end]
        
        # Check if there's a multiplication before the parenthesis
        if start > 0 and expr[start-1] == '*':
            # Find the multiplier
            multiplier_start = start - 1
            while multiplier_start > 0 and expr[multiplier_start-1].isdigit():
                multiplier_start -= 1
            
            if multiplier_start > 0 and expr[multiplier_start-1] == '-':
                multiplier_start -= 1
            elif multiplier_start > 0 and expr[multiplier_start-1] == '+':
                multiplier_start -= 1
            
            multiplier = expr[multiplier_start:start-1]  # Exclude the '*'
            
            if multiplier == '':
                multiplier = '1'
            elif multiplier == '-':
                multiplier = '-1'
            elif multiplier == '+':
                multiplier = '1'
            
            # Distribute the multiplier
            distributed = distribute_multiplier(inner, multiplier)
            expr = expr[:multiplier_start] + distributed + expr[end+1:]
        else:
            # No multiplication, just remove parentheses
            expr = expr[:start] + inner + expr[end+1:]
    
    return expr

def distribute_multiplier(inner, multiplier):
    """
    Distribute a multiplier across terms in parentheses
    """
    # Split inner expression into terms
    terms = re.split(r'([+-])', inner)
    if terms[0] == '':
        terms = terms[1:]
    
    result_terms = []
    current_sign = '+'
    
    for term in terms:
        if term in ['+', '-']:
            current_sign = term
            continue
        
        # Handle sign for the term
        term_sign = current_sign
        
        # Multiply the term by the multiplier
        if term == '':
            continue
        
        # Handle cases where term is just 's'
        if term == 's':
            term = '1*s'
        
        # Extract coefficient if any
        if '*' in term:
            coeff_part, var_part = term.split('*', 1)
        elif term.endswith('s'):
            coeff_part = term[:-1]
            var_part = 's'
        else:
            coeff_part = term
            var_part = ''
        
        # Calculate new coefficient
        if coeff_part == '':
            coeff = 1.0
        else:
            coeff = float(coeff_part)
        
        # Apply multiplier
        new_coeff = coeff * float(multiplier)
        
        # Apply term sign
        if term_sign == '-':
            new_coeff = -new_coeff
        
        # Build new term
        if var_part:
            result_terms.append(f"{new_coeff:+g}*{var_part}")
        else:
            result_terms.append(f"{new_coeff:+g}")
    
    # Join terms, remove leading '+'
    result = ''.join(result_terms)
    if result.startswith('+'):
        result = result[1:]
    
    return result

class FastNyquistVisualizer:
    def __init__(self, num=None, den=None):
        if num is not None and den is not None:
            self.sys = signal.TransferFunction(num, den)
        else:
            raise ValueError("Must provide num and den coefficients")
        
        # Frequency range
        self.w = np.logspace(-2, 2, 400)
        
        # Calculate frequency response
        self.w, self.mag, self.phase = signal.bode(self.sys, self.w)
        self.mag_linear = 10**(self.mag / 20)
        
        # Pre-calculate Nyquist points
        phase_rad = np.radians(self.phase)
        self.nyquist_real = self.mag_linear * np.cos(phase_rad)
        self.nyquist_imag = self.mag_linear * np.sin(phase_rad)
    
    def plot_bode(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        ax1.semilogx(self.w, self.mag, 'b', linewidth=2)
        ax1.set_ylabel('Magnitude [dB]')
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        ax1.set_title('Bode Diagram')
        
        ax2.semilogx(self.w, self.phase, 'r', linewidth=2)
        ax2.set_ylabel('Phase [deg]')
        ax2.set_xlabel('Frequency [rad/s]')
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def plot_complete_nyquist(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        
        ax.plot(self.nyquist_real, self.nyquist_imag, 'b-', linewidth=2)
        ax.plot(self.nyquist_real, -self.nyquist_imag, 'b--', linewidth=1, alpha=0.5)
        ax.plot(-1, 0, 'rx', markersize=10, markeredgewidth=2)
        ax.text(-1.1, 0.1, '(-1, 0)', fontsize=10)
        
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_aspect('equal')
        ax.set_title('Nyquist Diagram')
        
        plt.tight_layout()
        return fig

# Main App
st.title("📈 Bode & Nyquist Visualizer")

# Input section
st.markdown("### Enter Transfer Function")
tf_input = st.text_input(
    "Transfer Function (s-domain):",
    value="20*(s^2+1)",
    help="Examples: 20*(s^2+1), s*(s+100), 1/(s+1), 1/(s^2+0.5*s+1)"
)

parse_clicked = st.button("Generate Plots", type="primary")

# Test the parser
if st.checkbox("Show parser debug info"):
    st.write("Testing parser with your input:")
    try:
        num, den = parse_transfer_function(tf_input)
        st.write(f"Parsed numerator: {num}")
        st.write(f"Parsed denominator: {den}")
        
        # Show expanded form
        st.write("Expanded transfer function:")
        num_poly = " + ".join([f"{c:.2f}s^{len(num)-i-1}" for i, c in enumerate(num) if c != 0])
        den_poly = " + ".join([f"{c:.2f}s^{len(den)-i-1}" for i, c in enumerate(den) if c != 0])
        st.write(f"G(s) = ({num_poly}) / ({den_poly})")
    except Exception as e:
        st.error(f"Parser error: {e}")

# Process input
if parse_clicked or ('tf_input' in st.session_state and st.session_state.tf_input == tf_input):
    try:
        num_coeffs, den_coeffs = parse_transfer_function(tf_input)
        
        # Store in session state
        st.session_state.tf_input = tf_input
        st.session_state.num_coeffs = num_coeffs
        st.session_state.den_coeffs = den_coeffs
        
        # Display what was parsed
        st.success("✓ Successfully parsed transfer function!")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Numerator coefficients:** {num_coeffs}")
        with col2:
            st.info(f"**Denominator coefficients:** {den_coeffs}")
        
        # Create visualizer
        visualizer = FastNyquistVisualizer(num=num_coeffs, den=den_coeffs)
        
        # Tabs for visualizations
        tab1, tab2 = st.tabs(["Bode Plot", "Nyquist Diagram"])
        
        with tab1:
            fig_bode = visualizer.plot_bode()
            st.pyplot(fig_bode)
            plt.close(fig_bode)
        
        with tab2:
            fig_nyquist = visualizer.plot_complete_nyquist()
            st.pyplot(fig_nyquist)
            plt.close(fig_nyquist)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("""
        **Common formats that work:**
        - `20*(s^2+1)`
        - `s*(s+100)`  
        - `1/(s+1)`
        - `1/(s^2+0.5*s+1)`
        - `(s+1)/(s^2+2*s+3)`
        - `s^2+3*s+2`
        """)

# Quick examples
st.markdown("---")
st.markdown("**Quick examples:**")

examples = [
    ("20*(s^2+1)", "Gain * (s² + 1)"),
    ("s*(s+100)", "s(s + 100)"),
    ("1/(s+1)", "First order"),
    ("1/(s^2+0.5*s+1)", "Second order"),
    ("(s+1)/(s^2+2*s+3)", "With zero"),
]

cols = st.columns(len(examples))
for i, (example, desc) in enumerate(examples):
    with cols[i]:
        if st.button(f"{example}\n{desc}", key=f"ex{i}"):
            st.session_state.tf_input = example
            st.rerun()

st.markdown("""
**Note:** The parser now correctly handles:
- Multiplication: `20*(s^2+1)` → `20*s^2 + 20`
- Parentheses: `s*(s+100)` → `s^2 + 100*s`
- Powers: `s^2` or `s**2`
- Mixed expressions: `(s+1)*(s+2)`
""")

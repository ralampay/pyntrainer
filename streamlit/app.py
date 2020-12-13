import streamlit as st
import numpy as np
import pandas as pd

st.title('Autoencoder Visualization')

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])

st.line_chart(chart_data)

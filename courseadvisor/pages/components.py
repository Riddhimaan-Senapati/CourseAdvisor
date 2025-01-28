import streamlit as st
import plotly.graph_objects as go

def create_donut_chart(label,value, max_value=5.0):
    if value > 3.5:
        color = "green"
    elif 2 <= value <= 3.5:
        color = "yellow"
    else:
        color = "red"

    fig = go.Figure(data=[go.Pie(
        labels=['', label],
        values=[max_value - value, value],
        hole=.7,
        marker_colors=['#F0F0F0', color],
        textinfo='none',
        hoverinfo='none'
    )])

    fig.update_layout(
        annotations=[dict(text=f'{value:.1f}/{max_value:.1f}', x=0.5, y=0.5, font_size=20, showarrow=False)],
        showlegend=False,
        width=200,
        height=200,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    st.write(f"{label}")
    st.plotly_chart(fig, use_container_width=False)

def create_percentage_donut(label,percentage):
 
    if percentage > 70:
        color = "green"
    elif 40 <= percentage <= 70:
        color = "yellow"
    else:
        color = "red"

    fig = go.Figure(data=[go.Pie(
        labels=['', label],
        values=[100 - percentage, percentage],
        hole=.7,
        marker_colors=['#F0F0F0', color],
        textinfo='none',
        hoverinfo='none'
    )])

    fig.update_layout(
        annotations=[dict(text=f'{percentage:.1f}%', x=0.5, y=0.5, font_size=20, showarrow=False)],
        showlegend=False,
        width=200,
        height=200,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    st.write(f"{label}")
    st.plotly_chart(fig, use_container_width=False)
    

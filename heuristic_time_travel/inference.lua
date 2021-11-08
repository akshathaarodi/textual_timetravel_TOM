

import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#from inference.inference import Inference
model_name = "outputs-original/final_model.model" 
#inference_model = Inference(model_name)
model = torch.load(model_name, map_location='cuda:0')
def make_plot(num_cells, text, coref, overwrite, height=400, width=1000):
    fig = make_subplots(rows=num_cells, cols=1, # Grid config
                        shared_xaxes=True,  # Shared X-axis
                        vertical_spacing=0.02, y_title=r'Memory Cells',)

    # Find the global max and global min so that the heatmaps use the same boundaries
    zmin = min(np.min(overwrite), np.min(coref))
    zmax = max(np.max(overwrite), np.max(coref))

    for i in range(num_cells):
        # Add the heatmap
        fig.append_trace(
            go.Heatmap(
                z=np.stack([coref[i, :], overwrite[i, :]]),
                zmin=zmin - 0.3, # Offsetting the zmin because otherwise the lowest values become white
                zmax=zmax, showscale=False,  
                colorscale='Blues'
            ),
            row=i+1, col=1,  # Plot identification
        )
        # Set Y-axis ticks
        fig.update_yaxes(row=i+1, ticktext=[r'CR', r'OW'], showticklabels=True, 
                         tickcolor='#000000', tickvals=[0, 1], ticks='outside')

    # Set X-axis ticks shared by all the subplots
    fig.update_xaxes(row=num_cells, col=1, ticktext=text, showticklabels=True, tickvals=list(range(len(text))), 
                     tickangle=300, tickfont={'family':'Open Sans', 'size': 12}, tickcolor='#000000')

    # Update the layout and save the figure
    fig.update_layout(height=height, width=width, margin={'l': 60, 'r': 5, 't':5, 'b': 5})
    return fig


doc  = ("Amelia Shepherd, M.D. is a fictional character on the ABC American television medical drama Private Practice, and the spinoff series' progenitor show, Grey's Anatomy, "
        "portrayed by Caterina Scorsone. In her debut appearance in season three, Amelia visited her former sister-in-law, Addison Montgomery, and became a partner at the"
        " Oceanside Wellness Group.")
output = inference_model.perform_inference(doc)
# print(output.keys())

text = output["text"]
overwrite = np.array(output["overwrite"]).T  # num_cells x L
coref = np.array(output["coref"]).T  # num_cells x L

num_cells = overwrite.shape[0]
print (overwrite.shape, coref.shape) 


make_plot(num_cells, text, coref, overwrite, height=400, width=1000)

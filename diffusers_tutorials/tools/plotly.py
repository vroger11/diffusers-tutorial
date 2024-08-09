import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_generated_images(
    images: list,
    rows: int,
    cols: int,
    subplot_titles: list[str] = None,
    title_text: str = "Generated images",
) -> go.Figure:
    """Plot generated images inside a plotly subplot grid.

    Parameters
    ----------
    images : list
        images is a list of Pil Image.
    rows : int
        number of rows to use.
    cols : int
        number of columns to use.
    subplot_titles : list[str], optional
        List of title for each image, by default None

    Returns
    -------
        The Figure object associated
    """
    # Create subplots
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    # Add each image to the subplot
    for i, image in enumerate(images):
        row = i // cols + 1
        col = i % cols + 1
        # Convert the numpy array to a Plotly image
        fig.add_trace(go.Image(z=np.array(image)), row=row, col=col)

    # Update layout for better visualization
    fig.update_layout(
        height=rows * 300,
        width=cols * 300,
        title_text=title_text,
    )

    return fig

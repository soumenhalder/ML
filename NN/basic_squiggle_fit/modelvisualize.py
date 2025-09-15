# model_visualizer.py

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.layers import Dense
from prettytable import PrettyTable

class ModelVisualizer:
    """
    A utility class for visualizing and summarizing Keras Dense models.

    Features:
    - Tabular summary of all Dense layers (units, activation, weight/bias shapes)
    - Network structure visualization with neurons, connections, weights, and biases
    - Seaborn heatmaps of layer weights
    """

    def __init__(self, model):
        """
        Initialize the visualizer with a Keras model.

        Parameters:
        -----------
        model : tf.keras.Model
            The Keras model to visualize.
        """
        self.model = model
        self.layer_sizes = self._get_layer_sizes()
        self.weights_biases = self._extract_weights_biases()

    def _get_layer_sizes(self):
        """
        Get the number of neurons in each Dense layer including the input layer.

        Returns:
        --------
        layer_sizes : list[int]
            List containing neuron counts for input and Dense layers.
        """
        layer_sizes = [self.model.input_shape[-1]]
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                layer_sizes.append(layer.units)
        return layer_sizes

    def _extract_weights_biases(self):
        """
        Extract weights and biases from each Dense layer.

        Returns:
        --------
        extracted : list[tuple]
            List of tuples (weights, biases) for each Dense layer.
        """
        extracted = []
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                w, b = layer.get_weights() if layer.get_weights() else [None, None]
                extracted.append((w, b))
        return extracted

    def summarize_table(self):
        """
        Print a tabular summary of the model layers.

        Table includes:
        - Layer index and name
        - Layer type
        - Number of units
        - Activation function
        - Weight and bias shapes
        """
        table = PrettyTable()
        table.field_names = ["Layer", "Name", "Type", "Units", "Activation", "Weights Shape", "Bias Shape"]

        for idx, layer in enumerate(self.model.layers):
            if isinstance(layer, Dense):
                w, b = layer.get_weights() if layer.get_weights() else [None, None]
                table.add_row([
                    idx,
                    layer.name,
                    type(layer).__name__,
                    layer.units,
                    layer.activation.__name__,
                    w.shape if w is not None else None,
                    b.shape if b is not None else None
                ])
        print(table)

    def visualize_structure(self, show_weights=True):
        """
        Visualize the structure of the network with neurons, connections, weights, and biases.

        Parameters:
        -----------
        show_weights : bool, default=True
            Whether to display weight values on the connections.
        """
        h_spacing = 2
        total_height = max(self.layer_sizes) * 1.5

        fig, ax = plt.subplots(figsize=(max(8, len(self.layer_sizes)*2), 6))
        ax.axis('off')

        # Compute neuron coordinates
        coords = {}
        for layer_idx, layer_size in enumerate(self.layer_sizes):
            if layer_size > 1:
                y_positions = np.linspace(total_height/2, -total_height/2, layer_size)
            else:
                y_positions = [0]
            layer_coords = [(layer_idx*h_spacing, y) for y in y_positions]
            coords[layer_idx] = layer_coords
            for x, y in layer_coords:
                ax.add_patch(plt.Circle((x, y), 0.15, fc='lightblue', ec='black', zorder=3))

        # Draw connections, weights, and biases
        for l in range(len(self.layer_sizes)-1):
            w, b = self.weights_biases[l]
            Imax, Jmax = len(coords[l]), len(coords[l+1])
            for i, (x0, y0) in enumerate(coords[l]):
                for j, (x1, y1) in enumerate(coords[l+1]):
                    ax.plot([x0, x1], [y0, y1], color='gray', zorder=1)
                    if show_weights and w is not None:
                        delta = 0.5 * np.abs(Imax-i) + 1
                        mid_x = (x0 + delta * x1) / (1 + delta)
                        mid_y = (y0 + delta * y1) / (1 + delta)
                        angle = np.degrees(np.arctan2(y1 - y0, x1 - x0))
                        ax.text(mid_x, mid_y + 0.05, f'{w[i][j]:.2f}', fontsize=8,
                                rotation=angle, rotation_mode='anchor',
                                ha='left', va='bottom')
            # Draw biases
            if b is not None:
                for j, (x1, y1) in enumerate(coords[l+1]):
                    ax.text(x1, y1 + 0.25, f"b={b[j]:.2f}", fontsize=9, ha='center', color='darkred')

        ax.set_title("Neural Network Structure with Weights and Biases", fontsize=14)
        plt.show()

    def visualize_weights_heatmaps(self):
        """
        Plot Seaborn heatmaps of weights for each Dense layer.

        Each heatmap shows the weight matrix of a layer with neurons of the current
        layer on the y-axis and next layer neurons on the x-axis.
        """
        for i, (w, b) in enumerate(self.weights_biases):
            if w is not None:
                plt.figure(figsize=(6, 4))
                sns.heatmap(w, annot=False, cmap='coolwarm', cbar=True)
                plt.title(f"Weights Heatmap - Layer {i+1}")
                plt.xlabel("Neurons (next layer)")
                plt.ylabel("Neurons (current layer)")
                plt.show()


# model_visualizer_plotly.py

import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.layers import Dense
from prettytable import PrettyTable

class ModelVisualizerPlotly:
    """
    A utility class for visualizing and summarizing Keras Dense models using Plotly.

    Features:
    - Tabular summary of all Dense layers (units, activation, weight/bias shapes)
    - Interactive network structure visualization with neurons, connections, weights, and biases
    - Interactive heatmaps of layer weights
    """

    def __init__(self, model):
        """
        Initialize the visualizer with a Keras model.

        Parameters:
        -----------
        model : tf.keras.Model
            The Keras model to visualize.
        """
        self.model = model
        self.layer_sizes = self._get_layer_sizes()
        self.weights_biases = self._extract_weights_biases()

    def _get_layer_sizes(self):
        """Get number of neurons in each Dense layer including input layer."""
        layer_sizes = [self.model.input_shape[-1]]
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                layer_sizes.append(layer.units)
        return layer_sizes

    def _extract_weights_biases(self):
        """Extract weights and biases from each Dense layer."""
        extracted = []
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                w, b = layer.get_weights() if layer.get_weights() else [None, None]
                extracted.append((w, b))
        return extracted

    def summarize_table(self):
        """Print a tabular summary of the model layers."""
        table = PrettyTable()
        table.field_names = ["Layer", "Name", "Type", "Units", "Activation", "Weights Shape", "Bias Shape"]
        for idx, layer in enumerate(self.model.layers):
            if isinstance(layer, Dense):
                w, b = layer.get_weights() if layer.get_weights() else [None, None]
                table.add_row([
                    idx,
                    layer.name,
                    type(layer).__name__,
                    layer.units,
                    layer.activation.__name__,
                    w.shape if w is not None else None,
                    b.shape if b is not None else None
                ])
        print(table)

    def visualize_structure(self, show_weights=True):
        """Interactive network structure with Plotly."""
        h_spacing = 2
        total_height = max(self.layer_sizes) * 1.5

        coords = {}
        node_x, node_y, node_text = [], [], []
        edge_x, edge_y = [], []
        weight_x, weight_y, weight_text = [], [], []
        bias_x, bias_y, bias_text = [], [], []

        # Compute neuron coordinates
        for layer_idx, layer_size in enumerate(self.layer_sizes):
            y_positions = np.linspace(total_height/2, -total_height/2, layer_size) if layer_size > 1 else [0]
            coords[layer_idx] = [(layer_idx*h_spacing, y) for y in y_positions]
            for neuron_idx, (x, y) in enumerate(coords[layer_idx]):
                node_x.append(x)
                node_y.append(y)
                node_text.append(f"Layer {layer_idx}, Neuron {neuron_idx}")

        # Draw connections, weights, biases
        for l in range(len(self.layer_sizes)-1):
            w, b = self.weights_biases[l]
            Imax, Jmax = len(coords[l]), len(coords[l+1])
            for i, (x0, y0) in enumerate(coords[l]):
                for j, (x1, y1) in enumerate(coords[l+1]):
                    # Edge line
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]

                    # Weight text placement like matplotlib version
                    if show_weights and w is not None:
                        delta = 0.5 * np.abs(Imax - i) + 1
                        mid_x = (x0 + delta * x1) / (1 + delta)
                        mid_y = (y0 + delta * y1) / (1 + delta)
                        weight_x.append(mid_x)
                        weight_y.append(mid_y)
                        weight_text.append(f"{w[i][j]:.2f}")
            # Bias text above neurons
            if b is not None and show_weights:
                for j, (x1, y1) in enumerate(coords[l+1]):
                    bias_x.append(x1)
                    bias_y.append(y1 + 0.3)
                    bias_text.append(f"b={b[j]:.2f}")

        # Traces
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(width=1, color='gray'),
            hoverinfo='none'
        )

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            marker=dict(size=20, color='lightblue', line=dict(width=2, color='black')),
            text=node_text, hoverinfo='text'
        )

        weight_trace = go.Scatter(
            x=weight_x, y=weight_y, mode='text',
            text=weight_text,
            textfont=dict(size=8, color="red"),
            hoverinfo="none"
        )

        bias_trace = go.Scatter(
            x=bias_x, y=bias_y, mode='text',
            text=bias_text,
            textfont=dict(size=9, color="darkred"),
            hoverinfo="none"
        )

        fig = go.Figure(data=[edge_trace, node_trace, weight_trace, bias_trace])
        fig.update_layout(
            title="Interactive Neural Network Structure (with Weights & Biases)",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False)
        )
        fig.show()

    def visualize_weights_heatmaps(self):
        """Interactive Plotly heatmaps of each layer's weights."""
        for i, (w, b) in enumerate(self.weights_biases):
            if w is not None:
                fig = go.Figure(data=go.Heatmap(
                    z=w, colorscale="RdBu", colorbar=dict(title="Weight"),
                    hoverongaps=False
                ))
                fig.update_layout(
                    title=f"Weights Heatmap - Layer {i+1}",
                    xaxis_title="Next layer neurons",
                    yaxis_title="Current layer neurons"
                )
                fig.show()



# === Example usage ===
if __name__ == "__main__":
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense

    model = Sequential([
        Dense(4, activation='relu', input_shape=(3,), name='hidden_layer_1'),
        Dense(8, activation='relu', name='hidden_layer_2'),
        Dense(4, activation='relu', name='hidden_layer_3'),
        Dense(1, name='Output')
    ])
    #viz = ModelVisualizerPlotly(model)
    viz = ModelVisualizer(model)
    viz.summarize_table()
    viz.visualize_structure(show_weights=True)
    viz.visualize_weights_heatmaps()

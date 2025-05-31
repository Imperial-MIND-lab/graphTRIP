import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

# import matplotlib
# matplotlib.use('Agg') # Uncomment when running in debug mode

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_rel, linregress
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from nilearn import plotting, datasets, surface
from nilearn.image import new_img_like
import nibabel as nib


# Colourmap functions -----------------------------------------------------------

def custom_diverging_cmap(color1, color2, n_colors=256):
    """
    Create a custom diverging colormap with two input colors and white in the middle.
    
    Parameters:
    color1 (tuple): RGB values for the first color (0-1 range)
    color2 (tuple): RGB values for the second color (0-1 range)
    n_colors (int): Number of colors in the colormap
    
    Returns:
    matplotlib.colors.LinearSegmentedColormap: Custom diverging colormap
    """
    
    # Ensure n_colors is odd to have a perfect middle white
    if n_colors % 2 == 0:
        n_colors += 1
    
    # Create the colormap
    colors = []
    for i in range(n_colors):
        if i < n_colors // 2:
            r = np.interp(i, [0, n_colors // 2], [color1[0], 1])
            g = np.interp(i, [0, n_colors // 2], [color1[1], 1])
            b = np.interp(i, [0, n_colors // 2], [color1[2], 1])
        else:
            r = np.interp(i, [n_colors // 2, n_colors - 1], [1, color2[0]])
            g = np.interp(i, [n_colors // 2, n_colors - 1], [1, color2[1]])
            b = np.interp(i, [n_colors // 2, n_colors - 1], [1, color2[2]])
        colors.append((r, g, b))
    
    return LinearSegmentedColormap.from_list("custom_diverging", colors, N=n_colors)

def custom_sequential_cmap(color1, color2, n_colors=256):
    """
    Create a custom sequential colormap that gradually transitions from color1 to color2.
    
    Parameters:
    color1 (tuple): RGB values for the first color (0-1 range)
    color2 (tuple): RGB values for the second color (0-1 range)
    n_colors (int): Number of colors in the colormap
    
    Returns:
    matplotlib.colors.LinearSegmentedColormap: Custom sequential colormap
    """
    
    # Create the colormap
    colors = []
    for i in range(n_colors):
        r = np.interp(i, [0, n_colors - 1], [color1[0], color2[0]])
        g = np.interp(i, [0, n_colors - 1], [color1[1], color2[1]])
        b = np.interp(i, [0, n_colors - 1], [color1[2], color2[2]])
        colors.append((r, g, b))
    
    return LinearSegmentedColormap.from_list("custom_sequential", colors, N=n_colors)

def custom_multi_sequential_cmap(colors, n_colors=256):
    """
    Create a custom sequential colormap that gradually transitions through multiple input colors.
    
    Parameters:
    colors (list): List of RGB tuples in 0-1 range to transition through sequentially
    n_colors (int): Number of colors in the colormap
    
    Returns:
    matplotlib.colors.LinearSegmentedColormap: Custom sequential colormap
    """
    # Calculate colors per segment
    n_segments = len(colors) - 1  # number of transitions between colors
    colors_per_segment = n_colors // n_segments
    
    # Create the colormap
    output_colors = []
    
    # Generate colors for each segment
    for i in range(n_segments):
        start_color = colors[i]
        end_color = colors[i + 1]
        
        for j in range(colors_per_segment):
            t = j / colors_per_segment
            r = np.interp(t, [0, 1], [start_color[0], end_color[0]])
            g = np.interp(t, [0, 1], [start_color[1], end_color[1]])
            b = np.interp(t, [0, 1], [start_color[2], end_color[2]])
            output_colors.append((r, g, b))
    
    # Add the final color if we haven't reached n_colors
    if len(output_colors) < n_colors:
        output_colors.append(colors[-1])
    
    return LinearSegmentedColormap.from_list("custom_multi_sequential", output_colors, N=n_colors)

def custom_multi_diverging_cmap(colors1, colors2, n_colors=256):
    """
    Create a custom diverging colormap with multiple input colors on each side and white in the middle.
    Ensures symmetry and smooth transitions to white from both sides.
    
    Parameters:
    colors1 (list): List of RGB tuples for the left side (0-1 range)
    colors2 (list): List of RGB tuples for the right side (0-1 range)
    n_colors (int): Number of colors in the colormap
    
    Returns:
    matplotlib.colors.LinearSegmentedColormap: Custom diverging colormap
    """
    
    # Ensure n_colors is odd to have a perfect middle white
    if n_colors % 2 == 0:
        n_colors += 1
    
    # Calculate number of colors for each half
    half_colors = n_colors // 2
    
    # Create the colormap
    colors = []
    
    # Left side (colors1 to white)
    # Calculate total segments needed for the left side
    n_segments1 = len(colors1)  # including transition to white
    colors_per_segment1 = half_colors // n_segments1
    
    # Generate colors for each segment on the left side
    for i in range(n_segments1 - 1):
        start_color = colors1[i]
        end_color = colors1[i + 1]
        for j in range(colors_per_segment1):
            t = j / colors_per_segment1
            r = np.interp(t, [0, 1], [start_color[0], end_color[0]])
            g = np.interp(t, [0, 1], [start_color[1], end_color[1]])
            b = np.interp(t, [0, 1], [start_color[2], end_color[2]])
            colors.append((r, g, b))
    
    # Transition from last color1 to white
    start_color = colors1[-1]
    for j in range(colors_per_segment1):
        t = j / colors_per_segment1
        r = np.interp(t, [0, 1], [start_color[0], 1])
        g = np.interp(t, [0, 1], [start_color[1], 1])
        b = np.interp(t, [0, 1], [start_color[2], 1])
        colors.append((r, g, b))
    
    # Add middle white point
    colors.append((1, 1, 1))
    
    # Right side (white to colors2)
    n_segments2 = len(colors2)  # including transition from white
    colors_per_segment2 = half_colors // n_segments2
    
    # Transition from white to first color2
    end_color = colors2[0]
    for j in range(colors_per_segment2):
        t = j / colors_per_segment2
        r = np.interp(t, [0, 1], [1, end_color[0]])
        g = np.interp(t, [0, 1], [1, end_color[1]])
        b = np.interp(t, [0, 1], [1, end_color[2]])
        colors.append((r, g, b))
    
    # Generate colors for remaining segments on the right side
    for i in range(n_segments2 - 1):
        start_color = colors2[i]
        end_color = colors2[i + 1]
        for j in range(colors_per_segment2):
            t = j / colors_per_segment2
            r = np.interp(t, [0, 1], [start_color[0], end_color[0]])
            g = np.interp(t, [0, 1], [start_color[1], end_color[1]])
            b = np.interp(t, [0, 1], [start_color[2], end_color[2]])
            colors.append((r, g, b))
    
    return LinearSegmentedColormap.from_list("custom_multi_diverging", colors, N=n_colors)

def truncate_and_stretch_cmap(cmap, crange=(0, 1)):
    """
    Truncate and stretch a colormap to a new range.
    
    Parameters:
    cmap: matplotlib colormap
        The input colormap to modify
    crange: tuple (float, float)
        The new range as (min, max) where both values are between 0 and 1
        and min < max. For example, (0, 0.8) will use only the first 80% 
        of the colors and stretch them to the full range.
    
    Returns:
    matplotlib.colors.LinearSegmentedColormap: Modified colormap
    """
    if not (0 <= crange[0] < crange[1] <= 1):
        raise ValueError("crange values must be between 0 and 1 and min < max")
    
    # Get the original colormap colors
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)
    
    # Get n colors from original colormap in the specified range
    n_colors = 256  # or cmap.N for the original number of colors
    original_colors = cmap(np.linspace(crange[0], crange[1], n_colors))
    
    # Create new colormap with the truncated and stretched colors
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f"{cmap.name}_truncated",
        original_colors,
        N=n_colors)
    
    return new_cmap

def plot_colormap_stack(cmap, n_colors=10, make_plot=True):
    """
    Creates a horizontal stack of evenly spaced colors from a colormap.
    
    Parameters:
        cmap: matplotlib colormap
        n_colors (int): Number of colors to sample from the colormap
        make_plot (bool): Whether to make the plot
    """    
    # Get evenly spaced colors from the colormap
    if isinstance(cmap, str):
        cmap = sns.color_palette(cmap, as_cmap=True)
    colors = [cmap(i/(n_colors-1)) for i in range(n_colors)]

    if make_plot:
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(n_colors*0.4, 1))
        
        # Create horizontal stack
        left = 0
        width = 1/n_colors
        for color in colors:
            ax.barh(0, width, left=left, color=color, edgecolor='none')
            left += width
        
        # Customize plot
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlim(0, 1)
        
        # Add x tick labels with color indices
        ax.set_xticks([i/n_colors + width/2 for i in range(n_colors)])  # Center ticks in each segment
        ax.set_xticklabels(range(n_colors))
        ax.set_yticks([])
        
        # Remove frame
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()    
        plt.show()

    # Return the colors
    return colors

def plot_colorbar(cmap, vrange, orientation='horizontal', size=(6, 1), label=None, save_path=None):
    """
    Plot a standalone colorbar for a given colormap and value range.
    
    Parameters:
    -----------
    cmap : str or matplotlib.colors.Colormap
        Colormap to use for the colorbar
    vrange : tuple
        (vmin, vmax) tuple defining the range of values
    orientation : str, optional
        'horizontal' or 'vertical' orientation of the colorbar
    size : tuple, optional
        (width, height) of the figure in inches
    label : str, optional
        Label for the colorbar
    save_path : str, optional
        Path to save the figure. If None, figure is not saved
    """
    # Create figure and axis
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # [left, bottom, width, height]
    
    # Create a dummy mappable
    norm = mcolors.Normalize(vmin=vrange[0], vmax=vrange[1])
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # Create colorbar
    cbar = plt.colorbar(scalar_mappable, cax=ax, orientation=orientation)
    
    # Add label if provided
    if label:
        cbar.set_label(label)
    
    # Add black frame
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()

# Color constants -------------------------------------------------------------

# Diverging colormap
ylgnbu_colors = plot_colormap_stack('YlGnBu_r', 20, make_plot=False)
rocket_colors = plot_colormap_stack('rocket', 20, make_plot=False)
color1_indices = [1, 3, 5, 7]
color2_indices = [11, 9, 7, 5]
color1 = [ylgnbu_colors[i] for i in color1_indices]
color2 = [rocket_colors[i] for i in color2_indices]
COOLWARM = custom_multi_diverging_cmap(color1, color2)
COOLWARM_LIGHT = truncate_and_stretch_cmap(COOLWARM, (0.3, 0.7))

# Psilo and Escit colors
ylgnbu_colors = plot_colormap_stack('YlGnBu', 20, make_plot=False)
ESCIT = ylgnbu_colors[9]
PSILO = ylgnbu_colors[16]

# Receptor sequential colormaps
CMAP_5HT1A = sns.color_palette("YlGnBu", as_cmap=True)
CMAP_5HT2A = sns.color_palette("YlGnBu", as_cmap=True)
CMAP_5HTT = sns.color_palette("YlGnBu", as_cmap=True)

# Default colormap
CMAP_DEFAULT = sns.color_palette("YlGnBu", as_cmap=True)

# Alpha values
ALPHA_SCATTER = 0.8
BOX_COLOR = [0.9, 0.9, 0.9]

# Neutral colors
NEUTRAL = (0.8, 0.8, 0.8)                   # light gray
NEUTRAL2 = (0.4, 0.4, 0.4)                  # dark gray

# Plotting helpers ------------------------------------------------------------

def set_rsn_xticks(ax, rsn_mapping, rsn_labels):
    '''
    Orders the axis labels of the heatmap according to the RSNs.
    Parameters:
    -----------
    rsn_mapping (np.array): Array of indices indicating the RSNs for each ROI
    rsn_labels (list): List of RSN labels
    '''
    # Compute the tick positions and labels
    rsn_ticks = []
    rsn_tick_labels = []
    start = 0
    for i, rsn in enumerate(rsn_labels):
        end = np.sum(rsn_mapping == i) + start   # Find the range of rows/columns for this RSN
        rsn_ticks.append((start + end - 1) / 2)  # Place the label in the center
        rsn_tick_labels.append(rsn)
        start = end

    # Set the ticks and labels
    ax.set_xticks(rsn_ticks)
    ax.set_xticklabels(rsn_tick_labels, rotation=90)
    ax.set_yticks(rsn_ticks)
    ax.set_yticklabels(rsn_tick_labels)

def get_rsn_ticks(rsn_mapping, rsn_labels):
    '''
    Orders the axis labels of the heatmap according to the RSNs.
    Parameters:
    -----------
    rsn_mapping (np.array): Array of indices indicating the RSNs for each ROI
    rsn_labels (list): List of RSN labels
    '''
    # Compute the tick positions and labels
    rsn_ticks = []
    rsn_tick_labels = []
    start = 0
    for i, rsn in enumerate(rsn_labels):
        end = np.sum(rsn_mapping == i) + start   # Find the range of rows/columns for this RSN
        rsn_ticks.append((start + end - 1) / 2)  # Place the label in the center
        rsn_tick_labels.append(rsn)
        start = end

    return rsn_ticks, rsn_tick_labels

def get_atlas_img_size(atlas):
    '''
    Returns the size of the atlas image.
    '''
    if atlas.lower() == 'schaefer100':
        atlas_maps = datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=1)
        atlas_img = nib.load(atlas_maps.maps)  # Load the NIfTI file
        atlas_size = 100
    elif atlas.lower() == 'schaefer200':
        atlas_maps = datasets.fetch_atlas_schaefer_2018(n_rois=200, resolution_mm=2)
        atlas_img = nib.load(atlas_maps.maps)  # Load the NIfTI file
        atlas_size = 200
    elif atlas.lower() == 'aal':
        atlas_maps = datasets.fetch_atlas_aal()
        atlas_img = nib.load(atlas_maps.maps)  # Load the NIfTI file
        atlas_size = len(atlas_maps.labels)
    else:
        raise ValueError(f"Atlas {atlas} not yet implemented")

    return atlas_img, atlas_size

# Loss plotting -----------------------------------------------------------------

def plot_loss_curves(train_loss, test_loss, val_loss, save_path=None):
    '''Plots and saves the train, test and validation losses.'''
    num_folds = len(train_loss)
    num_rows = (num_folds + 2) // 3 
    fig, axes = plt.subplots(num_rows, 3, figsize=(12, 4*num_rows))
    axes = axes.flatten() 

    for k in range(num_folds):
        ax = axes[k]
        ax.plot(train_loss[k], color='blue', label='train')
        ax.plot(test_loss[k], color='red', label='test')
        if val_loss[k]:
            ax.plot(val_loss[k], color='orange', label='val')
        
        ax.set_title(f'fold {k+1}')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.grid(True)
        ax.legend(loc='upper right')

    # Turn off unused subplots
    for i in range(num_folds, len(axes)):
        axes[i].axis('off')

    if save_path is not None:
        format = save_path.split('.')[-1]
        plt.savefig(save_path, format=format)

# MLP plotting -----------------------------------------------------------------

def true_vs_pred_scatter(ypreds, marker_col=None, style=None, title=None, 
                         save_path=None, xcol='label', ycol='prediction'):
    """
    Create a scatter plot of true vs predicted values.
    
    Parameters:
    ypreds (pd.DataFrame): DataFrame containing columns for x and y values, and optionally 'Condition' column
    marker_col (str): Optional column name in ypreds to determine marker symbols
    style (str): Optional style for markers if 'Condition' column is not present. Can be 'psilo' or 'escit'
    title (str): Optional title for the plot
    save_path (str): Optional path to save the figure
    xcol (str): Name of column to use for x-axis values (default: 'label')
    ycol (str): Name of column to use for y-axis values (default: 'prediction')
    """
    plt.figure(figsize=(6, 5))

    # Define marker mapping if marker_col is provided
    if marker_col is not None:
        unique_markers = ypreds[marker_col].unique()
        marker_map = {val: marker for val, marker in 
                     zip(unique_markers, ['o', 'x', 'd', '^', 'v', '>', '<', 'p', 'h', '8'])}

    if 'Condition' in ypreds.columns:
        condition_psilo = ypreds[ypreds['Condition'] == 1.0]
        condition_escit = ypreds[ypreds['Condition'] == -1.0]
        
        # If marker_col is provided, create separate scatter plots for each marker type
        if marker_col is not None:
            for marker_val in ypreds[marker_col].unique():
                # Plot psilo points with this marker
                mask_psilo = condition_psilo[marker_col] == marker_val
                if mask_psilo.any():
                    plt.scatter(condition_psilo[mask_psilo][xcol], 
                              condition_psilo[mask_psilo][ycol],
                              marker=marker_map[marker_val], 
                              color=PSILO, edgecolor=PSILO, 
                              alpha=ALPHA_SCATTER)
                
                # Plot escit points with this marker
                mask_escit = condition_escit[marker_col] == marker_val
                if mask_escit.any():
                    plt.scatter(condition_escit[mask_escit][xcol], 
                              condition_escit[mask_escit][ycol],
                              marker=marker_map[marker_val], 
                              color=ESCIT, edgecolor=ESCIT, 
                              alpha=ALPHA_SCATTER)
        else:
            # Original behavior without marker_col
            plt.scatter(condition_psilo[xcol], condition_psilo[ycol], 
                      marker='d', color=PSILO, edgecolor=PSILO, alpha=ALPHA_SCATTER)
            plt.scatter(condition_escit[xcol], condition_escit[ycol], 
                      marker='o', color=ESCIT, edgecolor=ESCIT, alpha=ALPHA_SCATTER)
    else:
        if marker_col is not None:
            for marker_val in ypreds[marker_col].unique():
                mask = ypreds[marker_col] == marker_val
                color = PSILO if style == 'psilo' else ESCIT if style == 'escit' else NEUTRAL2
                plt.scatter(ypreds[mask][xcol], ypreds[mask][ycol],
                          marker=marker_map[marker_val], 
                          color=color, edgecolor=color, 
                          alpha=ALPHA_SCATTER)
        else:
            # Original behavior without marker_col
            color = PSILO if style == 'psilo' else ESCIT if style == 'escit' else NEUTRAL2
            marker = 'd' if style == 'psilo' else 'o'
            plt.scatter(ypreds[xcol], ypreds[ycol],
                      marker=marker, color=color, edgecolor=color, alpha=ALPHA_SCATTER)

    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.grid(True)
    
    min_val = min(ypreds[xcol].min(), ypreds[ycol].min())-2
    max_val = max(ypreds[xcol].max(), ypreds[ycol].max())+2
    plt.plot([min_val, max_val], [min_val, max_val], '--', color=NEUTRAL2, alpha=0.7)
    
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')
    
    if title is None:
        r, p = pearsonr(ypreds[xcol], ypreds[ycol])
        mae = np.mean(np.abs(ypreds[xcol] - ypreds[ycol]))
        mae_std = np.std(np.abs(ypreds[xcol] - ypreds[ycol]))
        title = f'r={r:.4f}, p={p:.4e}, MAE={mae:.4f} ± {mae_std:.4f}'
    plt.title(title)
    
    if save_path:
        format = save_path.split('.')[-1]
        plt.savefig(save_path, format=format)
    
    plt.show()

def true_vs_pred_scatter_with_patch(ypreds, marker_col=None, style=None, title=None, 
                         save_path=None, xcol='label', ycol='prediction',
                         patch_radius=None, patch_color='gray'):
    """
    Create a scatter plot of true vs predicted values.
    
    Parameters:
    ypreds (pd.DataFrame): DataFrame containing columns for x and y values, and optionally 'Condition' column
    marker_col (str): Optional column name in ypreds to determine marker symbols
    style (str): Optional style for markers if 'Condition' column is not present. Can be 'psilo' or 'escit'
    title (str): Optional title for the plot
    save_path (str): Optional path to save the figure
    xcol (str): Name of column to use for x-axis values (default: 'label')
    ycol (str): Name of column to use for y-axis values (default: 'prediction')
    patch_radius (float): Optional radius for shaded area around identity line
    patch_color (str): Color for the shaded area (default: 'gray')
    """
    plt.figure(figsize=(6, 5))

    # Define marker mapping if marker_col is provided
    if marker_col is not None:
        unique_markers = ypreds[marker_col].unique()
        marker_map = {val: marker for val, marker in 
                     zip(unique_markers, ['o', 'x', 'd', '^', 'v', '>', '<', 'p', 'h', '8'])}

    if 'Condition' in ypreds.columns:
        condition_psilo = ypreds[ypreds['Condition'] == 1.0]
        condition_escit = ypreds[ypreds['Condition'] == -1.0]
        
        # If marker_col is provided, create separate scatter plots for each marker type
        if marker_col is not None:
            for marker_val in ypreds[marker_col].unique():
                # Plot psilo points with this marker
                mask_psilo = condition_psilo[marker_col] == marker_val
                if mask_psilo.any():
                    plt.scatter(condition_psilo[mask_psilo][xcol], 
                              condition_psilo[mask_psilo][ycol],
                              marker=marker_map[marker_val], 
                              color=PSILO, edgecolor=PSILO, 
                              alpha=ALPHA_SCATTER)
                
                # Plot escit points with this marker
                mask_escit = condition_escit[marker_col] == marker_val
                if mask_escit.any():
                    plt.scatter(condition_escit[mask_escit][xcol], 
                              condition_escit[mask_escit][ycol],
                              marker=marker_map[marker_val], 
                              color=ESCIT, edgecolor=ESCIT, 
                              alpha=ALPHA_SCATTER)
        else:
            # Original behavior without marker_col
            plt.scatter(condition_psilo[xcol], condition_psilo[ycol], 
                      marker='d', color=PSILO, edgecolor=PSILO, alpha=ALPHA_SCATTER)
            plt.scatter(condition_escit[xcol], condition_escit[ycol], 
                      marker='o', color=ESCIT, edgecolor=ESCIT, alpha=ALPHA_SCATTER)
    else:
        if marker_col is not None:
            for marker_val in ypreds[marker_col].unique():
                mask = ypreds[marker_col] == marker_val
                color = PSILO if style == 'psilo' else ESCIT if style == 'escit' else NEUTRAL2
                plt.scatter(ypreds[mask][xcol], ypreds[mask][ycol],
                          marker=marker_map[marker_val], 
                          color=color, edgecolor=color, 
                          alpha=ALPHA_SCATTER)
        else:
            # Original behavior without marker_col
            color = PSILO if style == 'psilo' else ESCIT if style == 'escit' else NEUTRAL2
            marker = 'd' if style == 'psilo' else 'o'
            plt.scatter(ypreds[xcol], ypreds[ycol],
                      marker=marker, color=color, edgecolor=color, alpha=ALPHA_SCATTER)

    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.grid(True)
    
    min_val = min(ypreds[xcol].min(), ypreds[ycol].min())-2
    max_val = max(ypreds[xcol].max(), ypreds[ycol].max())+2
    
    # Create x values for the identity line and patch
    x = np.linspace(min_val, max_val, 100)
    
    # Plot the identity line
    plt.plot(x, x, '--', color=NEUTRAL2, alpha=0.7)
    
    # Add shaded area if patch_radius is provided
    if patch_radius is not None:
        plt.fill_between(x, x - patch_radius, x + patch_radius, 
                        color=patch_color, alpha=0.2)
    
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')
    
    if title is None:
        r, p = pearsonr(ypreds[xcol], ypreds[ycol])
        mae = np.mean(np.abs(ypreds[xcol] - ypreds[ycol]))
        mae_std = np.std(np.abs(ypreds[xcol] - ypreds[ycol]))
        title = f'r={r:.4f}, p={p:.4e}, MAE={mae:.4f} ± {mae_std:.4f}'
    plt.title(title)
    
    if save_path:
        format = save_path.split('.')[-1]
        plt.savefig(save_path, format=format)
    
    plt.show()

def regression_scatter(ypreds, marker_col=None, style=None, title=None, 
                      save_path=None, xcol='label', ycol='prediction',
                      show_ci=True, regline_alpha=0.6, equal_aspect=False, 
                      xlim=None, ylim=None):
    """
    Create a scatter plot with regression line and confidence intervals.
    
    Parameters:
    -----------
    ypreds (pd.DataFrame): DataFrame containing columns for x and y values, and optionally 'Condition' column
    marker_col (str): Optional column name in ypreds to determine marker symbols
    style (str): Optional style for markers if 'Condition' column is not present. Can be 'psilo' or 'escit'
    title (str): Optional title for the plot
    save_path (str): Optional path to save the figure
    xcol (str): Name of column to use for x-axis values (default: 'label')
    ycol (str): Name of column to use for y-axis values (default: 'prediction')
    show_ci (bool): Whether to show confidence intervals around regression line (default: True)
    regline_alpha (float): Alpha value for regression line (default: 0.6)
    """
    plt.figure(figsize=(6, 5))

    # Calculate correlation and p-value for title
    x = ypreds[xcol].values
    y = ypreds[ycol].values
    r_value, p_value = pearsonr(x, y)

    # Create base scatter plot with regression line and confidence intervals
    ax = sns.regplot(x=x, y=y, 
                    scatter=False,  # We'll add scatter points manually
                    line_kws={'color': 'darkred', 'alpha': regline_alpha},
                    ci=95 if show_ci else None)

    # Add scatter points with custom markers and colors
    if 'Condition' in ypreds.columns:
        condition_psilo = ypreds[ypreds['Condition'] == 1.0]
        condition_escit = ypreds[ypreds['Condition'] == -1.0]
        
        if marker_col is not None:
            unique_markers = ypreds[marker_col].unique()
            marker_map = {val: marker for val, marker in 
                         zip(unique_markers, ['o', 'x', 'd', '^', 'v', '>', '<', 'p', 'h', '8'])}
            
            for marker_val in ypreds[marker_col].unique():
                # Plot psilo points
                mask_psilo = condition_psilo[marker_col] == marker_val
                if mask_psilo.any():
                    ax.scatter(condition_psilo[mask_psilo][xcol], 
                             condition_psilo[mask_psilo][ycol],
                             marker=marker_map[marker_val], 
                             color=PSILO, edgecolor=PSILO, 
                             alpha=ALPHA_SCATTER)
                
                # Plot escit points
                mask_escit = condition_escit[marker_col] == marker_val
                if mask_escit.any():
                    ax.scatter(condition_escit[mask_escit][xcol], 
                             condition_escit[mask_escit][ycol],
                             marker=marker_map[marker_val], 
                             color=ESCIT, edgecolor=ESCIT, 
                             alpha=ALPHA_SCATTER)
        else:
            # Plot without marker_col
            ax.scatter(condition_psilo[xcol], condition_psilo[ycol], 
                      marker='d', color=PSILO, edgecolor=PSILO, alpha=ALPHA_SCATTER)
            ax.scatter(condition_escit[xcol], condition_escit[ycol], 
                      marker='o', color=ESCIT, edgecolor=ESCIT, alpha=ALPHA_SCATTER)
    else:
        if marker_col is not None:
            unique_markers = ypreds[marker_col].unique()
            marker_map = {val: marker for val, marker in 
                         zip(unique_markers, ['o', 'x', 'd', '^', 'v', '>', '<', 'p', 'h', '8'])}
            
            for marker_val in ypreds[marker_col].unique():
                mask = ypreds[marker_col] == marker_val
                color = PSILO if style == 'psilo' else ESCIT if style == 'escit' else NEUTRAL2
                ax.scatter(ypreds[mask][xcol], ypreds[mask][ycol],
                          marker=marker_map[marker_val], 
                          color=color, edgecolor=color, 
                          alpha=ALPHA_SCATTER)
        else:
            # Plot without marker_col
            color = PSILO if style == 'psilo' else ESCIT if style == 'escit' else NEUTRAL2
            marker = 'd' if style == 'psilo' else 'o'
            ax.scatter(ypreds[xcol], ypreds[ycol],
                      marker=marker, color=color, edgecolor=color, alpha=ALPHA_SCATTER)

    # Customize plot
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(True)
    
    # Add title
    if title is None:
        title = f'r={r_value:.4f}, p={p_value:.4e}'
    ax.set_title(title)

    # Scatter plot limits
    if equal_aspect and xlim is None and ylim is None:
        minval = min(ypreds[xcol].min(), ypreds[ycol].min())
        maxval = max(ypreds[xcol].max(), ypreds[ycol].max())
        padding = (maxval - minval) * 0.1
        ax.set_xlim(minval - padding, maxval + padding)
        ax.set_ylim(minval - padding, maxval + padding)
        ax.set_aspect('equal', adjustable='box')
    elif xlim is not None and ylim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    elif xlim is not None:
        ax.set_xlim(xlim)
        ymin = ypreds[ycol].min()
        ymax = ypreds[ycol].max()
        padding = (ymax - ymin) * 0.1
        ax.set_ylim(ymin - padding, ymax + padding)
    elif ylim is not None:
        ax.set_ylim(ylim)
        xmin = ypreds[xcol].min()
        xmax = ypreds[xcol].max() 
        padding = (xmax - xmin) * 0.1
        ax.set_xlim(xmin - padding, xmax + padding)
    else:
        xmin = ypreds[xcol].min()
        xmax = ypreds[xcol].max()
        ymin = ypreds[ycol].min()
        ymax = ypreds[ycol].max()
        xpadding = (xmax - xmin) * 0.1
        ypadding = (ymax - ymin) * 0.1
        ax.set_xlim(xmin - xpadding, xmax + xpadding)
        ax.set_ylim(ymin - ypadding, ymax + ypadding)
    
    if save_path:
        format = save_path.split('.')[-1]
        plt.savefig(save_path, format=format)
    
    plt.show()

def regression_scatter2(ypreds, title=None, 
                      save_path=None, xcol='label', ycol='prediction',
                      regline_alpha=0.6, equal_aspect=False, 
                      xlim=None, ylim=None, palette=None, symbols=None,
                      featcol='feature'):
    """
    Create a scatter plot with regression line.
    
    Parameters:
    -----------
    ypreds (pd.DataFrame): DataFrame containing columns for x and y values, and optionally 'Condition' column
    marker_col (str): Optional column name in ypreds to determine marker symbols
    title (str): Optional title for the plot
    save_path (str): Optional path to save the figure
    xcol (str): Name of column to use for x-axis values (default: 'label')
    ycol (str): Name of column to use for y-axis values (default: 'prediction')
    regline_alpha (float): Alpha value for regression line (default: 0.6)
    palette (dict): Optional dictionary mapping features to colors
    symbols (dict): Optional dictionary mapping features to marker symbols
    featcol (str): Name of column containing feature names (default: 'feature')
    """
    plt.figure(figsize=(6, 5))
    ax = plt.gca()

    # Calculate correlation and p-value for title
    x = ypreds[xcol].values
    y = ypreds[ycol].values
    r_value, p_value = pearsonr(x, y)

    # Compute and plot regression line manually
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.array([min(x), max(x)])
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, color='darkred', alpha=regline_alpha)

    # Add scatter points with custom markers and colors
    if 'Condition' in ypreds.columns:
        condition_psilo = ypreds[ypreds['Condition'] == 1.0]
        condition_escit = ypreds[ypreds['Condition'] == -1.0]
        
        if palette is not None:
            # Plot points using palette and symbols if provided
            for _, row in ypreds.iterrows():
                color = palette[row[featcol]]
                marker = symbols[row[featcol]] if symbols is not None else 'o'
                condition_color = PSILO if row['Condition'] == 1.0 else ESCIT
                plt.scatter(row[xcol], row[ycol], marker=marker, 
                          color=color, edgecolor=condition_color, alpha=ALPHA_SCATTER)
        else:
            # Plot without palette/symbols
            plt.scatter(condition_psilo[xcol], condition_psilo[ycol], 
                      marker='d', color=PSILO, edgecolor=PSILO, alpha=ALPHA_SCATTER)
            plt.scatter(condition_escit[xcol], condition_escit[ycol], 
                      marker='o', color=ESCIT, edgecolor=ESCIT, alpha=ALPHA_SCATTER)
    else:
        if palette is not None:
            # Plot points using palette and symbols if provided
            for _, row in ypreds.iterrows():
                color = palette[row[featcol]]
                marker = symbols[row[featcol]] if symbols is not None else 'o'
                plt.scatter(row[xcol], row[ycol], marker=marker,
                          color=color, alpha=ALPHA_SCATTER)
        else:
            # Plot without palette/symbols
            plt.scatter(ypreds[xcol], ypreds[ycol],
                      marker='o', color=NEUTRAL2, alpha=ALPHA_SCATTER)

    # Customize plot
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(True)
    
    # Add title
    if title is None:
        title = f'r={r_value:.4f}, p={p_value:.4e}'
    ax.set_title(title)

    # Scatter plot limits
    if equal_aspect and xlim is None and ylim is None:
        minval = min(ypreds[xcol].min(), ypreds[ycol].min())
        maxval = max(ypreds[xcol].max(), ypreds[ycol].max())
        padding = (maxval - minval) * 0.1
        ax.set_xlim(minval - padding, maxval + padding)
        ax.set_ylim(minval - padding, maxval + padding)
        ax.set_aspect('equal', adjustable='box')
    elif xlim is not None and ylim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    elif xlim is not None:
        ax.set_xlim(xlim)
        ymin = ypreds[ycol].min()
        ymax = ypreds[ycol].max()
        padding = (ymax - ymin) * 0.1
        ax.set_ylim(ymin - padding, ymax + padding)
    elif ylim is not None:
        ax.set_ylim(ylim)
        xmin = ypreds[xcol].min()
        xmax = ypreds[xcol].max() 
        padding = (xmax - xmin) * 0.1
        ax.set_xlim(xmin - padding, xmax + padding)
    else:
        xmin = ypreds[xcol].min()
        xmax = ypreds[xcol].max()
        ymin = ypreds[ycol].min()
        ymax = ypreds[ycol].max()
        xpadding = (xmax - xmin) * 0.1
        ypadding = (ymax - ymin) * 0.1
        ax.set_xlim(xmin - xpadding, xmax + xpadding)
        ax.set_ylim(ymin - ypadding, ymax + ypadding)
    
    if save_path:
        format = save_path.split('.')[-1]
        plt.savefig(save_path, format=format)
    
    plt.show()

def plot_legend(color_dict, orientation='horizontal', size=(6, 1), label=None, save_path=None):
    """
    Plot a standalone colorbar with discrete colors and labels from a dictionary.
    
    Parameters:
    -----------
    color_dict : dict
        Dictionary mapping strings (labels) to colors
    orientation : str, optional
        'horizontal' or 'vertical' orientation of the colorbar
    size : tuple, optional
        (width, height) of the figure in inches
    label : str, optional
        Label for the colorbar
    save_path : str, optional
        Path to save the figure. If None, figure is not saved
    """
    # Create figure and axis
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # [left, bottom, width, height]
    
    # Create custom colormap from dictionary
    colors = list(color_dict.values())
    labels = list(color_dict.keys())
    n_colors = len(colors)
    
    # Create boundaries and norm
    bounds = np.arange(n_colors + 1)
    norm = mcolors.BoundaryNorm(bounds, n_colors)
    
    # Create custom colormap
    cmap = mcolors.ListedColormap(colors)
    
    # Create scalar mappable
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # Create colorbar
    cbar = plt.colorbar(scalar_mappable, cax=ax, orientation=orientation,
                       ticks=bounds[:-1] + 0.5)  # Center ticks
    
    # Set tick labels
    if orientation == 'horizontal':
        cbar.ax.set_xticklabels(labels, rotation=90)
    else:
        cbar.ax.set_yticklabels(labels)
    
    # Add label if provided
    if label:
        cbar.set_label(label)
    
    # Add black frame
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()

# Dominance analysis plots ------------------------------------------------------

def permutation_importance_bar_chart(importance_scores, 
                                     y_column='mean', yerr_column='std', feature_column='feature',
                                     save_path=None, color=PSILO, alpha=0.5):
    """
    Creates a bar chart showing permutation importance scores with error bars.
    
    Parameters:
    ----------
    importance_scores : pandas.DataFrame
        DataFrame containing columns:
        - 'feature': string names for features (x-tick labels)
        - 'mean': mean importance scores (y-values)
        - 'std': standard deviation of importance scores (error bars)
    save_path : str, optional
        If provided, saves the figure to this path
    color : str or tuple, optional
        Color for the bars
    alpha : float, optional
        Alpha (transparency) value for the bars
    """
    # Create figure
    num_features = len(importance_scores)
    fig, ax = plt.subplots(1, 1, figsize=(num_features*0.5, 6))
    
    # Create bar chart
    bars = ax.bar(range(num_features), 
                  importance_scores[y_column], 
                  yerr=importance_scores[yerr_column], 
                  align='center',
                  color=color,
                  alpha=alpha)
    
    # Customize axes
    ax.set_ylabel('Importance Scores')
    ax.set_xticks(range(num_features))
    ax.set_xticklabels(importance_scores[feature_column], rotation=90)
    
    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax.set_axisbelow(True)  # Put grid lines behind bars
    
    if save_path:
        plt.savefig(save_path)

def plot_stacked_percentages(df, percentage_col, save_path=None, palette=None, figsize=(10, 4)):
    """
    Creates a stacked bar chart showing relative percentages for each receptor.
    
    Args:
        df (pd.DataFrame): DataFrame with receptor names as index and percentage column
        percentage_col (str): Name of column containing percentage values
        save_path (str, optional): Path to save the figure. Defaults to None.
        palette (list/dict, optional): Colors for each receptor. Defaults to None.
        figsize (tuple, optional): Figure size (width, height). Defaults to (10, 4).
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort values in descending order
    df_sorted = df.sort_values(percentage_col, ascending=True)
    
    # Get values and labels
    values = df_sorted[percentage_col].values
    labels = df_sorted.index
    
    # Create color palette if none provided
    if palette is None:
        palette = sns.color_palette("mako", len(df))
    elif isinstance(palette, str):
        palette = sns.color_palette(palette, len(df))

    # Create stacked bar
    left = 0
    for i, (value, label) in enumerate(zip(values, labels)):
        ax.barh(0, value, left=left, color=palette[i], label=label)
        
        # Add percentage labels in the middle of each segment
        if value >= 5:  # Only show label if segment is wide enough
            x_pos = left + value/2
            ax.text(x_pos, 0, f'{label}\n{value:.1f}%', 
                   ha='center', va='center')
        left += value
    
    # Customize plot
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel('Relative Importance (%)')
    
    # Remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)

def plot_stacked_percentages_v(df, percentage_col, save_path=None, palette=None, figsize=(4, 10)):
    """
    Creates a vertical stacked bar chart showing relative percentages for each receptor.
    
    Args:
        df (pd.DataFrame): DataFrame with receptor names as index and percentage column
        percentage_col (str): Name of column containing percentage values
        save_path (str, optional): Path to save the figure. Defaults to None.
        palette (list/dict, optional): Colors for each receptor. Defaults to None.
        figsize (tuple, optional): Figure size (width, height). Defaults to (4, 10).
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort values in descending order for bottom-to-top stacking
    df_sorted = df.sort_values(percentage_col, ascending=True)
    
    # Get values and labels
    values = df_sorted[percentage_col].values
    labels = df_sorted.index
    
    # Create color palette if none provided
    if palette is None:
        palette = sns.color_palette("mako", len(df))
    elif isinstance(palette, str):
        palette = sns.color_palette(palette, len(df))
    
    # Create stacked bar
    bottom = 0
    for i, (value, label) in enumerate(zip(values, labels)):
        ax.bar(0, value, bottom=bottom, color=palette[i], label=label)
        
        # Add percentage labels in the middle of each segment
        if value >= 5:  # Only show label if segment is tall enough
            y_pos = bottom + value/2
            ax.text(0, y_pos, f'{label}\n{value:.1f}%', 
                   ha='center', va='center')
        bottom += value
    
    # Customize plot
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_ylabel('Relative Importance (%)')
    
    # Remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)

# Distribution plots ------------------------------------------------------------

def plot_raincloud(distributions, palette=None, alpha=ALPHA_SCATTER, box_alpha=0.5, 
                   save_path=None, figsize=(8, 5), vline=None, xlim=None, sort_by_mean=False):
    """
    Creates a raincloud plot (half violin + points + boxplot) for multiple distributions.
    
    Parameters:
    -----------
    distributions : dict
        Dictionary where keys are distribution names and values are arrays of data
    palette : dict, optional
        Dictionary where keys are distribution names and values are colors
        If not provided, will use default color palette
    alpha : float, optional
        Transparency of violin plots and points (default: 0.5)
    box_alpha : float, optional
        Transparency of box plots (default: 0.3)
    save_path : str, optional
        If provided, saves the figure to this path
    figsize : tuple, optional
        Figure size in inches (default: (8, 6))
    add_vline : bool, optional
        If True, adds a vertical line at x=0
    xlim : tuple, optional
        Tuple of (min, max) for x-axis limits
    sort_by_mean : bool, optional
        If True, sorts distributions by their mean values (default: False)
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # If no palette provided, create one using default colors
    if palette is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(distributions)))
        palette = {name: color for name, color in zip(distributions.keys(), colors)}
    
    # Sort distributions by mean if requested
    if sort_by_mean:
        means = {name: np.mean(data) for name, data in distributions.items()}
        distributions = dict(sorted(distributions.items(), key=lambda x: means[x[0]], reverse=True))
        
    # Calculate y positions for each distribution
    y_positions = np.arange(len(distributions))
    
    # Add vertical line if requested
    if vline is not None:
        ax.axvline(x=vline, color='k', linestyle='--', linewidth=1, alpha=0.3, zorder=0)
    
    # Plot each distribution
    for idx, (name, data) in enumerate(distributions.items()):
        y_pos = y_positions[idx]
        color = palette.get(name)
        
        # Create violin plot
        violin_parts = ax.violinplot(data, positions=[y_pos], vert=False, 
                                   showmeans=False, showextrema=False)
        
        # Customize violin
        for pc in violin_parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(alpha)
            # Keep only right half
            m = np.mean([pc.get_paths()[0].vertices[:, 1].min(),
                        pc.get_paths()[0].vertices[:, 1].max()])
            pc.get_paths()[0].vertices[:, 1] = np.clip(pc.get_paths()[0].vertices[:, 1], m, np.inf)
        
        # Add jittered points
        y_jitter = np.random.normal(y_pos, 0.05, size=len(data))
        ax.scatter(data, y_jitter, color=color, alpha=alpha, s=20, zorder=2)
        
        # Add boxplot
        box_parts = ax.boxplot(data, positions=[y_pos], vert=False, widths=0.2,
                             showfliers=False, zorder=3, patch_artist=True)

        # Customize boxplot
        plt.setp(box_parts['boxes'], facecolor=color, alpha=box_alpha)  # Set face color and transparency
        plt.setp(box_parts['boxes'], edgecolor=color, linewidth=2)      # Set edge color without transparency
        plt.setp(box_parts['medians'], color=color, linewidth=2)
        plt.setp(box_parts['whiskers'], color=color, linewidth=2)
        plt.setp(box_parts['caps'], color=color, linewidth=2)
    
    # Customize plot
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(list(distributions.keys()))
    
    if xlim is not None:
        ax.set_xlim(xlim)
    
    ax.set_xlabel('Value')
    
    # Add some padding to y-axis
    ax.set_ylim(y_positions.min() - 0.5, y_positions.max() + 0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_diverging_raincloud(distributions, cmap=COOLWARM, vmax=None, 
                           alpha=0.5, box_alpha=0.5, scatter_alpha=0.3,
                           save_path=None, figsize=(8, 5), add_asterisk=False, add_colorbar=True):
    """
    Creates a diverging raincloud plot (half violin + points + boxplot) for multiple distributions.
    Boxplots and violins are in light gray, while scatter points are colored according to their values.
    
    Parameters:
    -----------
    distributions : dict
        Dictionary where keys are distribution names and values are arrays of data
    cmap : str or matplotlib colormap, optional
        Colormap to use for coloring the scatter points (default: 'coolwarm')
    vmax : float, optional
        Maximum absolute value for scaling the colormap. If None, uses max absolute value in data
    alpha : float, optional
        Transparency of violin plots (default: 0.5)
    box_alpha : float, optional
        Transparency of box plots (default: 0.5)
    scatter_alpha : float, optional
        Transparency of scatter points (default: 0.3)
    save_path : str, optional
        If provided, saves the figure to this path
    figsize : tuple, optional
        Figure size in inches (default: (8, 5))
    add_asterisk : bool, optional
        If True, adds asterisks for distributions significantly different from 0 (default: False)
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine vmax if not provided
    if vmax is None:
        vmax = max(abs(np.concatenate(list(distributions.values())).min()),
                  abs(np.concatenate(list(distributions.values())).max()))
    
    # Create colormap normalizer
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap(cmap)
    
    # Calculate y positions for each distribution
    y_positions = np.arange(len(distributions))
    
    # Add vertical line at 0
    ax.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.3, zorder=0)
    
    # If add_asterisk is True, perform t-tests
    if add_asterisk:
        pvalues = []
        means = []
        for data in distributions.values():
            t_stat, p_val = stats.ttest_1samp(data, 0)
            pvalues.append(p_val)
            means.append(np.median(data))
        
        # FDR correction
        significant_mask = fdrcorrection(pvalues)[0]
    
    # Plot each distribution
    for idx, (name, data) in enumerate(distributions.items()):
        y_pos = y_positions[idx]
        
        # Create violin plot in light gray
        violin_parts = ax.violinplot(data, positions=[y_pos], vert=False, 
                                   showmeans=False, showextrema=False)
        
        # Customize violin
        for pc in violin_parts['bodies']:
            pc.set_facecolor('lightgray')
            pc.set_alpha(alpha)
            # Keep only right half
            m = np.mean([pc.get_paths()[0].vertices[:, 1].min(),
                        pc.get_paths()[0].vertices[:, 1].max()])
            pc.get_paths()[0].vertices[:, 1] = np.clip(pc.get_paths()[0].vertices[:, 1], m, np.inf)
        
        # Add jittered points with colors based on values
        y_jitter = np.random.normal(y_pos, 0.05, size=len(data))
        scatter_colors = [cmap(norm(val)) for val in data]
        ax.scatter(data, y_jitter, c=scatter_colors, alpha=scatter_alpha, s=20, zorder=3)
        
        # Add boxplot in light gray
        box_parts = ax.boxplot(data, positions=[y_pos], vert=False, widths=0.2,
                             showfliers=False, zorder=2, patch_artist=True)

        # Customize boxplot
        plt.setp(box_parts['boxes'], facecolor='lightgray', alpha=box_alpha)
        plt.setp(box_parts['boxes'], edgecolor='gray', linewidth=1)
        plt.setp(box_parts['medians'], color='gray', linewidth=2)
        plt.setp(box_parts['whiskers'], color='gray', linewidth=1)
        plt.setp(box_parts['caps'], color='gray', linewidth=1)
        
        # Add asterisk if significant
        if add_asterisk and significant_mask[idx]:
            ax.text(means[idx], y_pos + 0.2, '*', color='darkred', 
                   ha='center', va='bottom', fontsize=12)
    
    # Customize plot
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(list(distributions.keys()))
    
    ax.set_xlabel('Value')
    
    # Add some padding to y-axis
    ax.set_ylim(y_positions.min() - 0.5, y_positions.max() + 0.5)
    
    # Add colorbar
    if add_colorbar:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Value')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
    
    return fig, ax

def plot_single_rain_violin(data, style_group=None, palette=None, symbols=None, alpha=ALPHA_SCATTER, 
                           save_path=None, figsize=(6, 8), hline=None, ylim=None, violin_side='left',
                           title=None):
    """
    Creates a single split violin plot with jittered points on the right side.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Array of data points to plot
    style_group : numpy.ndarray, optional
        Array assigning each data point to a style group. If None, all points use same style.
    palette : dict, optional
        Dictionary mapping style groups to colors. If None, uses default color palette.
    symbols : dict, optional
        Dictionary mapping style groups to marker symbols. If None, uses default markers.
    alpha : float, optional
        Transparency of violin plot and points (default: 0.5)
    save_path : str, optional
        If provided, saves the figure to this path
    figsize : tuple, optional
        Figure size in inches (default: (6, 8))
    hline : float, optional
        If provided, adds a horizontal line at this y-value
    ylim : tuple, optional
        Tuple of (min, max) for y-axis limits
    violin_side : str, optional
        Which side of the violin to show ('left' or 'right'). Default is 'left'.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # If no style groups provided, create a single group
    if style_group is None:
        style_group = np.zeros(len(data))
    
    # Get unique style groups
    unique_groups = np.unique(style_group)
    
    # Create default palette if none provided
    if palette is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
        palette = {group: color for group, color in zip(unique_groups, colors)}
    
    # Create default symbols if none provided
    if symbols is None:
        default_symbols = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'D']
        symbols = {group: default_symbols[i % len(default_symbols)] 
                  for i, group in enumerate(unique_groups)}
    
    # Create violin plot
    violin_parts = ax.violinplot(data, positions=[0], vert=True, 
                               showmeans=False, showextrema=False)
    
    # Customize violin
    for pc in violin_parts['bodies']:
        pc.set_facecolor('lightgray')
        pc.set_alpha(alpha)
        # Keep only specified half
        m = np.mean([pc.get_paths()[0].vertices[:, 0].min(),
                    pc.get_paths()[0].vertices[:, 0].max()])
        if violin_side == 'left':
            pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], -np.inf, m)
        else:  # right side
            pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], m, np.inf)
    
    # Add jittered points (opposite side of violin)
    x_jitter = np.random.normal(0, 0.02, size=len(data))  # Jitter around x=0.1
    
    for group in unique_groups:
        mask = style_group == group
        ax.scatter(x_jitter[mask], data[mask], 
                  color=palette[group], 
                  marker=symbols[group],
                  alpha=alpha, 
                  s=40, 
                  zorder=2)
    
    # Add horizontal line if requested
    if hline is not None:
        ax.axhline(y=hline, color='k', linestyle='--', linewidth=1, alpha=0.3, zorder=0)
    
    # Customize plot
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks([])
    if title is not None:
        ax.set_title(title)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    ax.set_ylabel('Value')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_histogram(distributions, palette=None, alpha=ALPHA_SCATTER, 
                   save_path=None, figsize=(8, 6), vline=None, xlim=None):
    """
    Creates a histogram plot with KDE lines for multiple distributions.
    
    Parameters:
    -----------
    distributions : dict
        Dictionary where keys are distribution names and values are arrays of data
    palette : dict, optional
        Dictionary where keys are distribution names and values are colors
        If not provided, will use default color palette
    alpha : float, optional
        Transparency of histogram bars (default: 0.5)
    save_path : str, optional
        If provided, saves the figure to this path
    figsize : tuple, optional
        Figure size in inches (default: (8, 6))
    vline : float, optional
        Vertical line to add to the plot
    xlim : tuple, optional
        Tuple of (min, max) for x-axis limits
    """
    # Create figure
    plt.figure(figsize=figsize)
    
    # If no palette provided, create one using default colors
    if palette is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(distributions)))
        palette = {name: color for name, color in zip(distributions.keys(), colors)}
    
    # Find common range for all distributions
    all_data = np.concatenate(list(distributions.values()))
    data_range = (np.min(all_data), np.max(all_data))

    # Add vertical line(s) if requested
    if vline is not None:
        if isinstance(vline, (list, np.ndarray)):
            for v in vline:
                plt.axvline(x=v, color='k', linestyle='--', linewidth=2, alpha=0.3)
        else:
            plt.axvline(x=vline, color='k', linestyle='--', linewidth=2, alpha=0.3)
    
    # Plot each distribution
    for name, data in distributions.items():
        color = palette.get(name)
        
        # Plot histogram
        plt.hist(data, bins='auto', density=True, alpha=alpha, 
                color=color, label=name)
        
        # Add KDE line
        kde = sns.kdeplot(data=data, color=color, linewidth=2)

    # Customize plot
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Density')
    if xlim is not None:
        plt.xlim(xlim)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_mean_bars(distributions, palette=None, figsize=(6, 7), box_alpha=0.5, scatter_alpha=0.7,
                  yline=None, ylabel='correlation (r)', save_path=None, add_scatter=False):
    """
    Create a bar plot showing mean performance with standard error bars for different models.
    
    Parameters:
    -----------
    distributions : dict
        Dictionary mapping model names to arrays of performance values
    palette : dict, optional
        Dictionary mapping model names to colors. If None, uses default colormap
    figsize : tuple, optional
        Figure size in inches (default: (6, 7))
    box_alpha : float, optional
        Alpha value for bar face color (default: 0.5)
    yline : float, optional
        Y-value for horizontal threshold line (e.g. minimum significant correlation)
    ylabel : str, optional
        Label for y-axis (default: 'correlation (r)')
    save_path : str, optional
        If provided, saves the figure to this path
    add_scatter : bool, optional
        If True, adds individual data points as scatter plot on top of bars (default: False)
    """
    # Generate default palette if none provided
    if palette is None:
        offset = 2
        colors = plot_colormap_stack('YlGnBu_r', len(distributions)+offset, make_plot=False)
        colors = colors[:-offset]
        palette = {name: color for name, color in zip(distributions.keys(), colors)}

    # Calculate means and standard errors
    means = {name: np.mean(values) for name, values in distributions.items()}
    sems = {name: np.std(values) / np.sqrt(len(values)) for name, values in distributions.items()}

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bars
    x = np.arange(len(distributions))
    bars = ax.bar(x, [means[name] for name in distributions.keys()],
                  yerr=[sems[name] for name in distributions.keys()],
                  capsize=5, zorder=2,
                  alpha=box_alpha)  
    
    # Set bar colors
    for bar, name in zip(bars, distributions.keys()):
        bar.set_facecolor((*plt.matplotlib.colors.to_rgb(palette[name]), box_alpha))
        bar.set_edgecolor(palette[name])

    # Add scatter points if requested
    if add_scatter:
        for i, (name, values) in enumerate(distributions.items()):
            # Add jitter to x-positions
            x_jitter = np.random.normal(i, 0.05, size=len(values))
            ax.scatter(x_jitter, values, 
                      color=palette[name],
                      alpha=scatter_alpha,
                      zorder=3)

    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.3, zorder=1)
    
    # Add threshold line if provided
    if yline is not None:
        ax.axhline(y=yline, color='red', linestyle='--', linewidth=1, zorder=3)

    # Customize plot
    ax.set_xticks(x)
    ax.set_xticklabels(distributions.keys(), rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig, ax

def plot_diverging_bars(df, yline=0, cmap=COOLWARM, vmax=None, alpha=0.5,
                       add_scatter=False, scatter_alpha=0.3, scatter_size=20,
                       figsize=None, save_path=None, add_colorbar=True):
    """
    Creates a diverging bar chart centered at yline.
    Each bar represents the mean of a column in the input DataFrame, with error bars
    showing standard error. If add_scatter is True, bars are light gray and scatter
    points are colored according to their values.
    
    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame where each column represents a feature, and each row represents an observation
    yline : float, optional
        The y-value at which the bars diverge (default: 0)
    cmap : str or matplotlib colormap, optional
        Colormap to use for coloring the scatter points (default: 'coolwarm')
    vmax : float, optional
        Maximum absolute value for scaling the colormap. If None, uses max absolute value in data
    alpha : float, optional
        Alpha (transparency) value for the bars
    add_scatter : bool, optional
        If True, adds scatter points showing individual data points
    scatter_alpha : float, optional
        Alpha value for scatter points (default: 0.3)
    scatter_size : float, optional
        Size of scatter points (default: 20)
    save_path : str, optional
        If provided, saves the figure to this path
    """
    # Create figure
    num_features = len(df.columns)
    if figsize is None:
        figsize = (num_features*0.75, 6)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Compute means and standard errors for each column
    means = df.mean()
    std_errors = df.sem()  # standard error of the mean
    
    # Determine vmax if not provided
    if vmax is None:
        vmax = max(abs(df.values.min()), abs(df.values.max()))
    
    # Create colormap normalizer
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap(cmap)
    
    # Create bars with colors based on values
    bars = []
    for i, (feature, mean) in enumerate(means.items()):
        # Use light gray for bars if scatter is enabled, otherwise use colormap
        bar_color = 'lightgray' if add_scatter else cmap(norm(mean))
        bar = ax.bar(i, mean - yline, bottom=yline, 
                    yerr=std_errors[feature],
                    color=bar_color, alpha=alpha,
                    capsize=0)  # Add error bar caps
        bars.append(bar)
        
        # Add scatter points if requested
        if add_scatter:
            # Add jitter to x-coordinates to prevent overlap
            x_jitter = np.random.normal(0, 0.1, len(df[feature]))
            # Color scatter points according to their values
            scatter_colors = [cmap(norm(val)) for val in df[feature]]
            ax.scatter(i + x_jitter, df[feature], 
                      c=scatter_colors, alpha=scatter_alpha,
                      s=scatter_size, zorder=3)  # zorder to ensure points are on top
    
    # Add dashed line at yline
    ax.axhline(y=yline, color='gray', linestyle='--', alpha=0.7)
    
    # Customize axes
    ax.set_ylabel('Value')
    ax.set_xticks(range(num_features))
    ax.set_xticklabels(df.columns, rotation=90)
    
    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax.set_axisbelow(True)  # Put grid lines behind bars
    
    # Add colorbar
    if add_colorbar:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Value')
    
    if save_path:
        plt.savefig(save_path)
    
    return fig, ax

def plot_simple_bars(values, feature_names=None, yline=0, cmap=COOLWARM, vmax=None, 
                            alpha=0.5, figsize=None, save_path=None, add_colorbar=True):
    """
    Creates a diverging bar chart centered at yline from an array of values.
    Each bar represents a value from the input array, colored according to its value.
    
    Parameters:
    ----------
    values : numpy.ndarray or list
        Array of values to plot as bars
    feature_names : list, optional
        Names for the x-axis tick labels. If None, uses indices
    yline : float, optional
        The y-value at which the bars diverge (default: 0)
    cmap : str or matplotlib colormap, optional
        Colormap to use for coloring the bars (default: 'coolwarm')
    vmax : float, optional
        Maximum absolute value for scaling the colormap. If None, uses max absolute value in data
    alpha : float, optional
        Alpha (transparency) value for the bars
    figsize : tuple, optional
        Figure size as (width, height). If None, automatically determined
    save_path : str, optional
        If provided, saves the figure to this path
    add_colorbar : bool, optional
        Whether to add a colorbar (default: True)
    """
    # Convert input to numpy array if it's a list
    values = np.array(values)
    
    # Create figure
    num_features = len(values)
    if figsize is None:
        figsize = (num_features*0.75, 6)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Determine vmax if not provided
    if vmax is None:
        vmax = max(abs(values.min()), abs(values.max()))
    
    # Create colormap normalizer
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap(cmap)
    
    # Create bars with colors based on values
    bars = []
    for i, value in enumerate(values):
        bar_color = cmap(norm(value))
        bar = ax.bar(i, value - yline, bottom=yline, 
                    color=bar_color, alpha=alpha)
        bars.append(bar)
    
    # Add dashed line at yline
    ax.axhline(y=yline, color='gray', linestyle='--', alpha=0.7)
    
    # Customize axes
    ax.set_ylabel('Value')
    ax.set_xticks(range(num_features))
    if feature_names is not None:
        ax.set_xticklabels(feature_names, rotation=90)
    
    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax.set_axisbelow(True)  # Put grid lines behind bars
    
    # Add colorbar
    if add_colorbar:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Value')
    
    if save_path:
        plt.savefig(save_path)
    
    return fig, ax

# Brain surface plotting --------------------------------------------------------

def plot_brain_surface(data, 
                       atlas='schaefer100', 
                       threshold=None, 
                       cmap='jet', 
                       vrange=None, 
                       title=None, 
                       save_path=None,
                       mesh='fsaverage5'):
    """
    Plot parcellated data on brain surface views.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Data array of shape (num_nodes, 1) or (num_nodes,) containing values for each parcel
    atlas : str
        Name of the atlas to use ('schaefer100', 'aal', etc.)
    threshold : float, optional
        Threshold for displaying values
    cmap : str, optional
        Matplotlib colormap name
    vrange : tuple, optional
        (vmin, vmax) tuple defining the range of the colormap. If None, will use data min/max.
    title : str, optional
        Title for the plot
    save_path : str, optional
        File path to save the figure. If None, figure is not saved.
    mesh : str, optional
        Mesh to use for plotting. Defaults to 'fsaverage'.
    """
    # Load appropriate atlas
    atlas_img, atlas_size = get_atlas_img_size(atlas)
    region_indices = np.unique(atlas_img.get_fdata()) # for AAL, they are not simply 0-num_regions!

    # Rest of the function remains the same
    data = np.asarray(data).ravel()
    if len(data) != atlas_size:
        raise ValueError(f"Data length {len(data)} doesn't match atlas size ({atlas_size})")
    
    # Create volume from parcellated data
    atlas_data = atlas_img.get_fdata()
    volume_data = np.zeros_like(atlas_data)
    
    # Map parcel values to volume
    for i, value in enumerate(data, 1):
        volume_data[atlas_data == region_indices[i]] = value
    
    volume_img = new_img_like(atlas_img, volume_data)
    
    # Load surface for plotting
    fsaverage = datasets.fetch_surf_fsaverage(mesh=mesh)
    
    # If vrange not provided, compute from data
    if vrange is None:
        vrange = (np.min(data), np.max(data))
        
    # Create figure with 4 subplots (2 views x 2 hemispheres)
    fig = plt.figure(figsize=(12, 8))
    views = ['lateral', 'medial']
    hemispheres = ['left', 'right']
    
    # Create subplots for each view and hemisphere combination
    for i, view in enumerate(views):
        for j, hemi in enumerate(hemispheres):
            # Calculate subplot position (2 rows, 2 columns)
            pos = i * 2 + j + 1
            ax = fig.add_subplot(2, 2, pos, projection='3d')
            
            pial_mesh = fsaverage[f'pial_{hemi}']
            infl_mesh = fsaverage[f'infl_{hemi}']
            
            texture = surface.vol_to_surf(
                volume_img, 
                pial_mesh,
                interpolation='nearest')
            
            plotting.plot_surf_stat_map(
                infl_mesh,
                stat_map=texture,
                hemi=hemi,
                view=view,
                colorbar=True if (hemi == 'right' and view == 'medial') else False,
                threshold=threshold,
                axes=ax,
                bg_map=None,
                cmap=cmap,
                vmin=vrange[0],
                vmax=vrange[1])
            
            # Add subplot title
            ax.set_title(f'{hemi.capitalize()} - {view.capitalize()}')
    
    if title:
        fig.suptitle(title, y=0.95, fontsize=14)
        
    if save_path:
        plt.savefig(save_path)
        
    return fig

def plot_brain_surface_grid(data, atlas='schaefer100', threshold=None, 
                            cmap='jet', column_names=None, view='medial', 
                            vranges=None, save_path=None,
                            mesh='fsaverage5'):
    """
    Plot multiple rows of parcellated data on brain surface views.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Data array of shape (num_rows, num_nodes, n_cols) where:
        - num_rows is the number of rows to plot
        - num_nodes is the number of brain regions
        - n_cols is the number of columns/features to plot
    atlas : str
        Name of the atlas to use ('schaefer100', 'aal', etc.)
    threshold : float, optional
        Threshold for displaying values
    cmap : str or list, optional
        Either a single matplotlib colormap name or a list of colormaps,
        one for each column
    column_names : list, optional
        List of names for each column
    view : str, optional
        Brain view to plot ('lateral' or 'medial')
    vranges : list, optional
        List of tuples (vmin, vmax) for each column
    save_path : str, optional
        File path to save the figure
    """
    # Input validation
    data = np.asarray(data)
    if len(data.shape) != 3:
        raise ValueError("data must be a 3D array (num_rows, num_nodes, n_cols)")
    
    num_rows, num_nodes, n_cols = data.shape
    n_cols = min(n_cols, 5)  # Limit to 5 columns
    
    # Load atlas and get region indices
    atlas_img, atlas_size = get_atlas_img_size(atlas)
    region_indices = np.unique(atlas_img.get_fdata()) # for AAL, they are not simply 0-num_regions!

    # Load surface for plotting
    fsaverage = datasets.fetch_surf_fsaverage(mesh=mesh)
    
    # Convert single cmap to list if necessary
    if isinstance(cmap, (str, mcolors.Colormap)):
        cmaps = [cmap] * n_cols
    else:
        if len(cmap) < n_cols:
            raise ValueError(f"If providing a list of colormaps, must provide at least {n_cols} cmaps")
        cmaps = cmap[:n_cols]
    
    # Create figure with extra width for colorbars
    fig = plt.figure(figsize=(3*n_cols, 3*num_rows))
    
    # Process each column
    for col in range(n_cols):
        if vranges is not None:
            vrange = vranges[col]
        else:
            # Calculate shared vrange for this column across all rows using 95th percentile
            vmin = np.percentile(data[:, :, col], 5)
            vmax = np.percentile(data[:, :, col], 95)
            vrange = (vmin, vmax)
        
        # Process each row
        for row in range(num_rows):
            # Create volume from parcellated data
            volume_data = np.zeros_like(atlas_img.get_fdata())
            for i, value in enumerate(data[row, :, col], 1):
                volume_data[atlas_img.get_fdata() == region_indices[i]] = value
            volume_img = new_img_like(atlas_img, volume_data)
            
            # Create subplot
            ax = fig.add_subplot(num_rows, n_cols, row * n_cols + col + 1, projection='3d')
            
            for hemi in ['left', 'right']:
                pial_mesh = fsaverage[f'pial_{hemi}']
                infl_mesh = fsaverage[f'infl_{hemi}']
                
                texture = surface.vol_to_surf(
                    volume_img, 
                    pial_mesh,
                    interpolation='nearest')
                
                plotting.plot_surf_stat_map(
                    infl_mesh,
                    stat_map=texture,
                    hemi=hemi,
                    view=view,
                    colorbar=True if hemi == 'right' else False,
                    threshold=threshold,
                    axes=ax,
                    bg_map=None,
                    cmap=cmaps[col],  # Use the appropriate colormap for this column
                    vmin=vrange[0],
                    vmax=vrange[1])
            
            # Remove axis labels
            ax.set_axis_off()
            
            # Add column labels only to top row
            if row == 0 and column_names is not None:
                ax.set_title(column_names[col], pad=10)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        
    return fig

# Attention weight RSN plots ---------------------------------------------------

def plot_rsn_attention_boxplots(mean_rsn_attention, context_attr_name='Condition', 
                                palette='Set2', alpha=0.5, marker='o', save_path=None):
    """
    Creates boxplots comparing attention weights across RSNs for different conditions.
    Excludes outliers beyond 95th percentile from scatter points.
    
    Args:
        mean_rsn_attention (pd.DataFrame): DataFrame with columns for each RSN, plus 'Condition' and 'Subject'
        context_attr_name (str): Name of the condition column (default='Condition')
        palette (str/dict): Color palette or dict mapping conditions to colors
        alpha (float): Alpha value for scatter points
        marker (str): Marker style for scatter points
        save_path (str, optional): Path to save the figure
        
    Returns:
        pd.DataFrame: Statistics DataFrame with test results
    """
    # Get RSN names (all columns except Condition and Subject)
    rsn_names = [col for col in mean_rsn_attention.columns 
                 if col not in [context_attr_name, 'Subject']]
    context_values = mean_rsn_attention[context_attr_name].unique()
    
    # Create figure and axes
    fig, axes = plt.subplots(1, len(rsn_names), figsize=(20, 5))
    if len(rsn_names) == 1:
        axes = [axes]
    
    # Convert palette to dict if string
    if isinstance(palette, str):
        colors = sns.color_palette(palette, n_colors=len(context_values))
        palette = dict(zip(context_values, colors))

    if isinstance(marker, str):
        markers = [marker]*len(context_values)
        marker_dict = dict(zip(context_values, markers))
    else:
        marker_dict = marker
    
    # Statistical tests setup
    stats_results = []
    pvals = []
    
    for rsn_idx, (rsn_name, ax) in enumerate(zip(rsn_names, axes)):
        # Create boxplot with light gray boxes
        sns.boxplot(data=mean_rsn_attention, x=context_attr_name, y=rsn_name,
                   color=BOX_COLOR, showfliers=False, width=0.5,
                   ax=ax, zorder=1)
        
        # Calculate 95th percentile for y-axis limit
        y_95th = np.percentile(mean_rsn_attention[rsn_name], 95)
        ax.set_ylim(0-0.01*y_95th, y_95th*1.25)
        
        # Add jittered points below 95th percentile
        for context_val in context_values:
            context_data = mean_rsn_attention[mean_rsn_attention[context_attr_name] == context_val]
            mask = context_data[rsn_name] <= y_95th
            x_pos = list(context_values).index(context_val)
            x_jitter = np.random.normal(x_pos, 0.05, size=len(context_data))
            ax.scatter(x_jitter[mask], context_data[rsn_name][mask],
                      color=palette[context_val], alpha=alpha, marker=marker_dict[context_val], zorder=2)
        
        # Paired t-test
        data_cond1 = mean_rsn_attention[mean_rsn_attention[context_attr_name] == context_values[0]][rsn_name]
        data_cond2 = mean_rsn_attention[mean_rsn_attention[context_attr_name] == context_values[1]][rsn_name]
        t_stat, p_val = ttest_rel(data_cond1, data_cond2)
        
        stats_results.append({
            'RSN': rsn_name,
            'T-statistic': t_stat,
            'p-value': p_val})
        pvals.append(p_val)
        
        # Customize subplot
        ax.set_title(f'{rsn_name}\np={p_val:.3f}')
        if rsn_idx != 0:
            ax.set_ylabel('')

    # FDR correction and add asterisks
    _, pvals_corrected = fdrcorrection(pvals, alpha=0.05)
    
    for ax, p_corr in zip(axes, pvals_corrected):
        if p_corr < 0.05:
            y_max = ax.get_ylim()[1]
            ax.text(0.5, y_max*0.90, '*', color='red', ha='center', fontsize=20)
        title = ax.get_title()
        ax.set_title(f'{title}\nfdr={p_corr:.3f}')

    plt.tight_layout()
    
    # Create stats DataFrame
    stats_df = pd.DataFrame(stats_results)
    stats_df['fdr-corrected'] = pvals_corrected
    
    if save_path:
        plt.savefig(save_path)
        
    return stats_df

def plot_rsn_attention_boxplots_connected(mean_rsn_attention, context_attr_name='Condition', 
                                          palette='Set2', alpha=0.5, marker='o', save_path=None):
    """
    Creates boxplots with all data points shown and connected by lines for paired samples.
    
    Args:
        mean_rsn_attention (pd.DataFrame): DataFrame with columns for each RSN, plus 'Condition' and 'Subject'
        context_attr_name (str): Name of the condition column (default='Condition')
        palette (str/dict): Color palette or dict mapping conditions to colors
        alpha (float): Alpha value for scatter points
        marker (str): Marker style for scatter points
        save_path (str, optional): Path to save the figure
        
    Returns:
        pd.DataFrame: Statistics DataFrame with test results
    """
    # Get RSN names and context values
    rsn_names = [col for col in mean_rsn_attention.columns 
                 if col not in [context_attr_name, 'Subject']]
    context_values = mean_rsn_attention[context_attr_name].unique()
    
    # Create figure and axes
    fig, axes = plt.subplots(1, len(rsn_names), figsize=(20, 5))
    if len(rsn_names) == 1:
        axes = [axes]
    
    # Convert palette to dict if string
    if isinstance(palette, str):
        colors = sns.color_palette(palette, n_colors=len(context_values))
        palette = dict(zip(context_values, colors))

    if isinstance(marker, str):
        markers = [marker]*len(context_values)
        marker_dict = dict(zip(context_values, markers))
    else:
        marker_dict = marker
    
    # Statistical tests setup
    stats_results = []
    pvals = []
    
    for rsn_idx, (rsn_name, ax) in enumerate(zip(rsn_names, axes)):
        # Create boxplot with light gray boxes
        sns.boxplot(data=mean_rsn_attention, x=context_attr_name, y=rsn_name,
                   color=BOX_COLOR, showfliers=False, width=0.5,
                   ax=ax, zorder=1)
        
        # Add connected points for each subject
        for subject in mean_rsn_attention['Subject'].unique():
            subject_data = mean_rsn_attention[mean_rsn_attention['Subject'] == subject]
            if len(subject_data) == 2:  # Ensure we have both conditions
                x_jitter = np.random.normal(0, 0.05)
                y_values = [subject_data[subject_data[context_attr_name] == val][rsn_name].iloc[0] 
                          for val in context_values]
                
                # Draw connecting line
                ax.plot([0 + x_jitter, 1 + x_jitter], y_values,
                       color='gray', alpha=0.7, zorder=1, linewidth=0.5)
                
                # Draw points
                for i, (x, y) in enumerate(zip([0 + x_jitter, 1 + x_jitter], y_values)):
                    ax.scatter(x, y, color=palette[context_values[i]], 
                             alpha=alpha, marker=marker_dict[context_values[i]], zorder=2)
        
        # Paired t-test
        data_cond1 = mean_rsn_attention[mean_rsn_attention[context_attr_name] == context_values[0]][rsn_name]
        data_cond2 = mean_rsn_attention[mean_rsn_attention[context_attr_name] == context_values[1]][rsn_name]
        t_stat, p_val = ttest_rel(data_cond1, data_cond2)
        
        stats_results.append({
            'RSN': rsn_name,
            'T-statistic': t_stat,
            'p-value': p_val
        })
        pvals.append(p_val)
        
        # Customize subplot
        ax.set_title(f'{rsn_name}\np={p_val:.4f}')
        if rsn_idx != 0:
            ax.set_ylabel('')
            
        # Set y-axis limits with margin
        y_min = mean_rsn_attention[rsn_name].min()
        y_max = mean_rsn_attention[rsn_name].max()
        margin = 0.05 * (y_max - y_min)
        ax.set_ylim(y_min - margin, y_max + margin)

    # FDR correction and add asterisks
    _, pvals_corrected = fdrcorrection(pvals, alpha=0.05)
    
    for ax, p_corr in zip(axes, pvals_corrected):
        if p_corr < 0.05:
            y_max = ax.get_ylim()[1]
            ax.text(0.5, y_max*0.90, '*', color='red', ha='center', fontsize=20)
        title = ax.get_title()
        ax.set_title(f'{title}\nfdr={p_corr:.4f}')

    plt.tight_layout()
    
    # Create stats DataFrame
    stats_df = pd.DataFrame(stats_results)
    stats_df['fdr-corrected'] = pvals_corrected
    
    if save_path:
        plt.savefig(save_path)
        
    return stats_df

# Gradient alignment plots -----------------------------------------------------

def plot_clustered_matrix(mat, labels, testfold_indices=None, features=None, 
                          vrange=None, cmap=COOLWARM, figsize=(15, 10), save_path=None):
    """
    Plot alignment matrix with samples sorted by cluster membership.
    
    Parameters:
    -----------
    mat (numpy.ndarray): Matrix to plot (num_samples, num_features)
    labels (numpy.ndarray): Cluster labels for each sample
    subjects (array-like, optional): Subject IDs. If None, uses indices
    testfold_indices (array-like, optional): Test fold indices for each subject
    features (list, optional): Feature names for x-axis. If None, uses indices
    vrange (tuple, optional): (min, max) values for color scaling
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Sort subjects by cluster label
    cluster_order = np.argsort(labels)
    sorted_mat = mat[cluster_order]
    
    # Create default values if not provided
    subjects = np.arange(len(labels))
    if features is None:
        features = [f'F{i}' for i in range(mat.shape[1])]
    if vrange is None:
        # Scale to 95th percentile of absolute values
        max_abs_val = np.percentile(np.abs(sorted_mat), 95)
        vrange = (-max_abs_val, max_abs_val)
    
    # Create y-tick labels that show cluster membership
    unique_clusters = np.unique(labels)
    if testfold_indices is not None:
        y_labels = [f'Sub {subjects[i]} (C{labels[i]}, fold {testfold_indices[i]})' 
                   for i in cluster_order]
    else:
        y_labels = [f'Sub {subjects[i]} (C{labels[i]})' for i in cluster_order]
    
    # Create the plot
    fig = plt.figure(figsize=figsize)
    sns.heatmap(sorted_mat, 
                cmap=cmap,
                vmin=vrange[0], 
                vmax=vrange[1],
                center=0,
                xticklabels=features,
                yticklabels=y_labels)
    plt.xticks(rotation=90)
    
    # Add horizontal lines to separate clusters
    current_pos = 0
    for cluster in unique_clusters:
        current_pos += np.sum(labels[cluster_order] == cluster)
        if current_pos < len(labels):  # No line after last cluster
            plt.axhline(y=current_pos, color='black', linewidth=2)
    
    plt.title('Mean Alignment by Subject and Feature\nGrouped by Cluster')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
def plot_clustered_matrix2(mat, labels, features=None, conditions=None,
                          vrange=None, cmap=COOLWARM, figsize=(15, 10), save_path=None):
    """
    Plot alignment matrix with samples sorted by cluster membership and condition.
    
    Parameters:
    -----------
    mat (numpy.ndarray): Matrix to plot (num_samples, num_features)
    labels (numpy.ndarray): Cluster labels for each sample
    features (list, optional): Feature names for x-axis. If None, uses indices
    conditions (array-like, optional): Condition labels for each sample
    vrange (tuple, optional): (min, max) values for color scaling
    cmap (str, optional): Colormap to use
    figsize (tuple, optional): Figure size
    save_path (str, optional): Path to save figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # First sort by cluster label, then by condition within each cluster
    subjects = np.arange(len(labels))
    if conditions is not None:
        # Create array of indices sorted first by cluster then condition
        cluster_condition_order = []
        for cluster in np.unique(labels):
            cluster_mask = labels == cluster
            cluster_indices = np.where(cluster_mask)[0]
            # Sort these indices by condition
            cluster_conditions = conditions[cluster_indices]
            condition_order = np.argsort(cluster_conditions)
            cluster_condition_order.extend(cluster_indices[condition_order])
        sorted_mat = mat[cluster_condition_order]
        sorted_labels = labels[cluster_condition_order]
        sorted_conditions = conditions[cluster_condition_order]
        sorted_subjects = subjects[cluster_condition_order]
    else:
        # Just sort by cluster if no conditions provided
        cluster_order = np.argsort(labels)
        sorted_mat = mat[cluster_order]
        sorted_labels = labels[cluster_order]
        sorted_subjects = subjects[cluster_order]
        sorted_conditions = None
    
    # Create default values if not provided
    if features is None:
        features = [f'F{i}' for i in range(mat.shape[1])]
    if vrange is None:
        # Scale to 95th percentile of absolute values
        max_abs_val = np.percentile(np.abs(sorted_mat), 95)
        vrange = (-max_abs_val, max_abs_val)
    
    # Create y-tick labels showing cluster membership and condition
    if sorted_conditions is not None:
        y_labels = [f'Sub {s}, {c}' 
                   for s, c in zip(sorted_subjects, sorted_conditions)]
    else:
        y_labels = [f'Sub {s}' for s in sorted_subjects]
    
    # Create the plot
    fig = plt.figure(figsize=figsize)
    sns.heatmap(sorted_mat, 
                cmap=cmap,
                vmin=vrange[0], 
                vmax=vrange[1],
                center=0,
                xticklabels=features,
                yticklabels=y_labels)
    plt.xticks(rotation=90)
    
    # Add horizontal lines to separate clusters
    current_pos = 0
    for cluster in np.unique(sorted_labels):
        current_pos += np.sum(sorted_labels == cluster)
        if current_pos < len(labels):  # No line after last cluster
            plt.axhline(y=current_pos, color='black', linewidth=2)
    
    plt.title('Mean Alignment by Subject and Feature\nGrouped by Cluster')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)

def plot_heatmap(mat, features=None, vrange=None, cmap=COOLWARM, figsize=(15, 10), save_path=None):
    """
    Plot alignment matrix as a heatmap.
    
    Parameters:
    -----------
    mat (numpy.ndarray): Matrix to plot (num_samples, num_features)
    features (list, optional): Feature names for x-axis. If None, uses indices
    vrange (tuple, optional): (min, max) values for color scaling
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Create default values if not provided
    if features is None:
        features = [f'F{i}' for i in range(mat.shape[1])]
    if vrange is None:
        # Scale to 95th percentile of absolute values
        max_abs_val = np.percentile(np.abs(mat), 95)
        vrange = (-max_abs_val, max_abs_val)
    
    # Create y-tick labels
    y_labels = [f'Sub{i}' for i in range(mat.shape[0])]
    
    # Create the plot
    fig = plt.figure(figsize=figsize)
    sns.heatmap(mat, 
                cmap=cmap,
                vmin=vrange[0], 
                vmax=vrange[1],
                center=0,
                xticklabels=features,
                yticklabels=y_labels)
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)

# VGAE plotting -----------------------------------------------------------------

def plot_fc_reconstructions(adj_orig_rcn, fold_assignments, 
                           rsn_mapping=None, rsn_labels=None, 
                           cmap=COOLWARM, vrange=None, save_path=None):
    '''
    Plots FC matrices as heatmaps, one example per fold.
    
    Parameters:
    ----------
    adj_orig_rcn: Dict with 'original' and 'reconstructed' arrays of shape 
                  (n_nodes, n_nodes, n_samples) for adjacency matrices
    fold_assignments: Array of length n_samples indicating which fold's VGAE was used
    rsn_mapping: Optional mapping from ROI labels to RSNs
    rsn_labels: Optional labels for RSNs
    cmap: Colormap for FC matrix plots
    vrange: Optional value range for colormap
    save_path: Optional path to save plots
    '''
    rsn_indices = None
    if rsn_mapping is not None and rsn_labels is not None:
        rsn_indices = np.argsort(rsn_mapping)
    
    k = int(np.max(fold_assignments) + 1)
    
    # Plot one example from each fold
    fig, axes = plt.subplots(1, k, figsize=(5*k, 4))
    if k == 1:
        axes = [axes]
        
    for fold in range(k):
        # Get first example from this fold
        fold_samples = np.where(fold_assignments == fold)[0]
        example_idx = fold_samples[0]

        # Get reconstructed (triu) and original (tril) matrices and combine
        orig = adj_orig_rcn['original'][:, :, example_idx]
        recon = adj_orig_rcn['reconstructed'][:, :, example_idx]
        if rsn_indices is not None:
            orig = orig[rsn_indices][:, rsn_indices]
            recon = recon[rsn_indices][:, rsn_indices]
        L = np.tril(orig)
        U = np.triu(recon, 1)
        matrix = L + U
            
        # Plot
        if vrange is not None:
            im = axes[fold].imshow(matrix, cmap=cmap, aspect='equal', 
                                   vmin=vrange[0], vmax=vrange[1])
        else:
            im = axes[fold].imshow(matrix, cmap=cmap, aspect='equal')
            
        if rsn_indices is not None and rsn_labels is not None:
            set_rsn_xticks(axes[fold], rsn_mapping, rsn_labels)
            
        axes[fold].set_title(f'fold {fold+1}: orig (L) vs. recon (U)')
        plt.colorbar(im, ax=axes[fold])
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_fc_reconstruction_single(adj_orig_rcn, subject_idx, 
                                  rsn_mapping=None, rsn_labels=None, 
                                  cmap=COOLWARM, vrange=None, save_path=None):
    '''
    Plots FC matrix as heatmap for a single subject, showing original vs reconstructed.
    
    Parameters:
    ----------
    adj_orig_rcn: Dict with 'original' and 'reconstructed' arrays of shape 
                  (n_nodes, n_nodes, n_samples) for adjacency matrices
    subject_idx: Integer indicating which subject/sample to plot
    rsn_mapping: Optional mapping from ROI labels to RSNs
    rsn_labels: Optional labels for RSNs
    cmap: Colormap for FC matrix plots
    vrange: Optional value range for colormap
    save_path: Optional path to save plot
    '''
    rsn_indices = None
    if rsn_mapping is not None and rsn_labels is not None:
        rsn_indices = np.argsort(rsn_mapping)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    
    # Get reconstructed (triu) and original (tril) matrices and combine
    orig = adj_orig_rcn['original'][:, :, subject_idx]
    recon = adj_orig_rcn['reconstructed'][:, :, subject_idx]
    if rsn_indices is not None:
        orig = orig[rsn_indices][:, rsn_indices]
        recon = recon[rsn_indices][:, rsn_indices]
    L = np.tril(orig)
    U = np.triu(recon, 1)
    matrix = L + U
        
    # Plot
    if vrange is not None:
        im = ax.imshow(matrix, cmap=cmap, aspect='equal', 
                      vmin=vrange[0], vmax=vrange[1])
    else:
        im = ax.imshow(matrix, cmap=cmap, aspect='equal')
        
    if rsn_indices is not None and rsn_labels is not None:
        set_rsn_xticks(ax, rsn_mapping, rsn_labels)
        
    ax.set_title('original (L) vs. reconstructed (U)')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_fc_correlations(fc_corrs, conditions=None, save_path=None):
    '''
    Makes boxplot of correlations between original and reconstructed matrices.
    
    Parameters:
    ----------
    fc_corrs: Array of shape (n_samples) containing the FC correlations
    conditions: Array of length n_samples indicating which condition each sample belongs to
    save_path: Optional path to save plot
    '''
    # Make boxplot
    if not np.isnan(fc_corrs).any():
        fig, ax = plt.subplots(1, 1, figsize=(2, 5))
        # Set zorder=1 for boxplot to ensure it's behind the scatter points
        sns.boxplot(fc_corrs, ax=ax, color=BOX_COLOR, showfliers=False, zorder=1)
        ax.set_ylabel('FC correlation')
        ax.set_ylim(min(min(fc_corrs)-0.05, 0.7), min(max(fc_corrs)+0.05, 1))

        # Add individual points with jitter and zorder=2 to ensure they're on top
        x_jitter = np.random.normal(0, 0.05, size=len(fc_corrs))
        if conditions is not None:
            for sub, fc_corr in enumerate(fc_corrs):
                if conditions[sub] == -1:
                    marker, color = 'o', ESCIT
                elif conditions[sub] == 1:
                    marker, color = 'd', PSILO
                else:
                    marker, color = 'o', NEUTRAL2
                ax.scatter(x_jitter[sub], fc_corr, marker=marker, 
                        color=color, alpha=ALPHA_SCATTER, zorder=2)
        else:
            ax.scatter(x_jitter, fc_corrs, marker='o', 
                      color=NEUTRAL2, alpha=ALPHA_SCATTER, zorder=2)

        # Adjust layout to prevent y-label cutoff
        plt.tight_layout()
        plt.subplots_adjust(left=0.3)

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_metric_boxplot(df, conditions=None, short_names=True, ylabel='MAE', save_path=None):
    """
    Creates a boxplot of metric values with optional condition-based scatter points.
    
    Parameters:
    -----------
    df (pd.DataFrame): DataFrame containing values to plot
    conditions (array-like, optional): Array of conditions (-1, 1, or other) for each sample
    short_names (bool): If True, uses shortened column names (first part before '_')
    ylabel (str): Label for y-axis
    save_path (str, optional): Path to save the figure
    """
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(len(df.columns)*0.8+1, 7))
    
    # Create boxplot
    sns.boxplot(data=df, ax=ax, color=BOX_COLOR, showfliers=False, zorder=1)
    
    # Add scatter points
    for i, col in enumerate(df.columns):
        x_jitter = np.random.normal(i, 0.05, size=len(df))
        if conditions is not None:
            for sub, val in enumerate(df[col]):
                if conditions[sub] == -1:
                    marker, color = 'o', ESCIT
                elif conditions[sub] == 1:
                    marker, color = 'd', PSILO
                else:
                    marker, color = 'o', NEUTRAL2
                ax.scatter(x_jitter[sub], val, marker=marker, color=color, alpha=ALPHA_SCATTER, zorder=2)
        else:
            ax.scatter(x_jitter, df[col], color=NEUTRAL2, alpha=ALPHA_SCATTER, zorder=2)
    
    # Set labels
    if short_names:
        column_names = [col.split('_')[0] for col in df.columns]
    else:
        column_names = df.columns
    ax.set_xticks(range(len(column_names)))
    ax.set_xticklabels(column_names, rotation=90, ha='center')
    ax.set_ylabel(ylabel)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_x_reconstructions(x_orig_rcn, atlas=None, conditions=None, short_names=True, save_path=None, only_boxplots=False):
    '''Plots the normalized MAE of node embeddings.
    Parameters:
    ----------
    x_orig_rcn (dict): Output from get_test_reconstructions
        keys: 'original' (n_nodes, n_features, n_samples), 
              'reconstructed' (n_nodes, n_features, n_samples), 
              'feature_names' (n_features)
    atlas (str): Atlas name
    conditions (np.array): Array of length n_samples indicating which condition each sample belongs to
    short_names (bool): Whether to use short names for the columns
    save_path (str): Optional path to save plot
    '''
    # Fetch metrics
    metrics = x_orig_rcn['metrics']
    mae_df = metrics['mae']
    corr_df = metrics['corr']
    save_paths = []
    if save_path:
        parts = save_path.rsplit('.', 1)

    # Plot MAE as boxplot
    if not only_boxplots:
        save_path_mae = None
        if save_path:
            save_path_mae = f"{parts[0]}_mae.{parts[1]}"
            save_paths.append(save_path_mae)
        plot_metric_boxplot(mae_df, conditions=conditions, 
                            short_names=short_names, 
                            ylabel='MAE', save_path=save_path_mae)
    
    # Plot correlations as boxplot
    save_path_corr = None
    if save_path:
        save_path_corr = f"{parts[0]}_corr.{parts[1]}"
        save_paths.append(save_path_corr)
    plot_metric_boxplot(corr_df, conditions=conditions, 
                        short_names=short_names, 
                        ylabel='Correlation', save_path=save_path_corr)
    
    # Plot brain surface
    if not only_boxplots:
        save_path_surf = None
        if save_path:
            save_path_surf = f"{parts[0]}_surf.{parts[1]}"
        subject_idx = 0
        try:
            data2plot = np.stack([x_orig_rcn['reconstructed'][:, :, subject_idx], 
                                  x_orig_rcn['original'][:, :, subject_idx]], axis=0)
            plot_brain_surface_grid(data2plot, 
                                    atlas=atlas, 
                                    view='medial',
                                    column_names=x_orig_rcn['feature_names'],
                                    save_path=save_path_surf)
            save_paths.append(save_path_surf)
        except Exception as e:
            print(f'Skipping brain surface plot. Error: {str(e)}')
            save_path_surf = None

    return save_paths

def plot_checkpoint_metrics(metrics_dict, conditions=None, save_path=None):
    """
    Creates boxplots for FC and node feature metrics across checkpoints.
    
    Parameters:
    -----------
    metrics_dict (dict): Dictionary containing:
        - 'fc_corr': DataFrame with checkpoints as columns, samples as rows
        - 'fc_mae': DataFrame with checkpoints as columns, samples as rows
        - 'node_corr': Dict of DataFrames, one per node feature
    conditions (array-like, optional): Array of conditions (-1, 1, or other) for each sample
    save_dir (str, optional): Directory to save the figures
    
    Returns:
    --------
    dict: Dictionary of generated figures
    """
    save_paths = []
    if save_path:
        parts = save_path.rsplit('.', 1)
    
    # Plot FC correlations
    save_path_fc_corr = None
    if save_path:
        save_path_fc_corr = f"{parts[0]}_fc_corr.{parts[1]}"
        save_paths.append(save_path_fc_corr)
    plot_metric_boxplot(
        metrics_dict['fc_corr'], 
        conditions=conditions,
        ylabel='correlation',
        save_path=save_path_fc_corr,
        short_names=False)
    
    # Plot FC MAE
    save_path_fc_mae = None
    if save_path:
        save_path_fc_mae = f"{parts[0]}_fc_mae.{parts[1]}"
        save_paths.append(save_path_fc_mae)
    plot_metric_boxplot(
        metrics_dict['fc_mae'],
        conditions=conditions,
        ylabel='MAE',
        save_path=save_path_fc_mae,
        short_names=False)
    
    # Plot node feature correlations
    for feature, df in metrics_dict['node_corr'].items():
        save_path_node_corr = None
        if save_path:
            save_path_node_corr = f"{parts[0]}_node_corr_{feature}.{parts[1]}"
            save_paths.append(save_path_node_corr)
        plot_metric_boxplot(df,
            conditions=conditions,
            ylabel=f'{feature} correlation',
            save_path=save_path_node_corr,
            short_names=False)
    
    return save_paths

def plot_vgae_reconstructions(adj_orig_rcn, 
                              x_orig_rcn, 
                              fold_assignments,
                              conditions=None,
                              cmap=COOLWARM, 
                              rsn_mapping=None, 
                              rsn_labels=None,
                              atlas=None,
                              vrange=None, 
                              save_path=None,
                              only_boxplots=False):
    '''Wrapper function that calls the three separate plotting functions.'''
    save_paths = []
    if save_path:
        save_path_parts = save_path.rsplit('.', 1)
    
    if not only_boxplots:
        # Plot matrices
        save_path_recon = None
        if save_path:
            save_path_recon = f"{save_path_parts[0]}_fc_recon.{save_path_parts[1]}"
            save_paths.append(save_path_recon)
        plot_fc_reconstructions(adj_orig_rcn, fold_assignments, 
                                rsn_mapping=rsn_mapping, rsn_labels=rsn_labels,
                                cmap=cmap, vrange=vrange, save_path=save_path_recon)
    
    # Plot correlations
    save_path_stats = None
    if save_path:
        save_path_stats = f"{save_path_parts[0]}_fc_corrs.{save_path_parts[1]}"
    try:   
        plot_fc_correlations(adj_orig_rcn['metrics']['corr'], conditions, save_path=save_path_stats)
        save_paths.append(save_path_stats)
    except Exception as e:
        print(f'Skipping FC correlation plot. Error: {str(e)}')
        save_path_stats = None
    
    # Plot MAE of node embeddings
    if x_orig_rcn:
        save_path_x = None
        if save_path:
            save_path_x = f"{save_path_parts[0]}_x.{save_path_parts[1]}"
        save_paths += plot_x_reconstructions(x_orig_rcn, conditions=conditions, 
                                            atlas=atlas, save_path=save_path_x,
                                            only_boxplots=only_boxplots)

    return save_paths


# Interpretability ------------------------------------------------------------

def plot_attention_weights(attention_df, grouping_col=None):
    """
    Plot mean attention weights across brain regions, optionally grouped by a categorical variable.
    
    Parameters:
    -----------
    attention_df : pandas.DataFrame
        DataFrame containing attention weights and grouping variables
    grouping_col : str, optional
        Name of column to group by. If None, plots single graph of all data.
    """
    if grouping_col is None:
        # Single plot for all data
        means = attention_df.mean()
        stds = attention_df.std()
        
        fig, ax = plt.subplots(1, 1, figsize=(20, 6))
        ax.bar(range(len(means)), means, yerr=stds, capsize=2)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(means.index, rotation=90)
        ax.set_ylabel('Mean Attention Weight')
        ax.set_title('Mean Attention Weights by Brain Region')
        
    else:
        # Get unique values in grouping column
        groups = attention_df[grouping_col].unique()
        
        # Create subplot for each group
        fig, axes = plt.subplots(len(groups), 1, figsize=(20, 6*len(groups)))
        if len(groups) == 1:
            axes = [axes]
            
        for i, group_val in enumerate(groups):
            # Get data for this group
            group_data = attention_df[attention_df[grouping_col] == group_val].drop(grouping_col, axis=1)
            means = group_data.mean()
            stds = group_data.std()
            
            # Create bar plot
            axes[i].bar(range(len(means)), means, yerr=stds, capsize=2)
            axes[i].set_xticks(range(len(means)))
            axes[i].set_xticklabels(means.index, rotation=90)
            axes[i].set_ylabel('Mean Attention Weight')
            axes[i].set_title(f'Mean Attention Weights by Brain Region ({grouping_col} = {group_val})')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

# Classification ----------------------------------------------------------------

def plot_confusion_matrix(predictions, labels, threshold=0.5, save_path=None):
    """Plot confusion matrix for binary classification.
    
    Args:
        predictions: Array-like of predicted probabilities
        labels: Array-like of true labels (0 or 1)
        threshold: Classification threshold for predictions (default 0.5)
        save_path: Optional path to save figure
    """
    # Create confusion matrix
    y_pred = (predictions > threshold).astype(int)
    y_true = labels.astype(int)
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap='Blues')

    # Add numbers to cells
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j],
                          ha="center", va="center", color="black")

    # Add colorbar
    plt.colorbar(im)

    # Add labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Escitalopram', 'Psilocybin'])
    ax.set_yticklabels(['Escitalopram', 'Psilocybin'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Add title
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_roc_curve(predictions, labels, figsize=(4, 4), save_path=None):
    """Plot ROC curve for binary classifier predictions.
    
    Args:
        predictions: Array of predicted probabilities
        labels: Array of true binary labels
        save_path: Optional path to save figure
    """
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(labels, predictions)
    auc_score = roc_auc_score(labels, predictions)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='blue', lw=1, 
            label=f'ROC (AUC = {auc_score:.2f})')
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    
    # Customize plot
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    # ax.set_aspect('equal')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
    plt.show()

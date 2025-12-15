import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def plot_sample_field(
    sample_coords,
    sample_u,
    sample_v,
    sample_pressure,
    boundary_coords,
    mask,
    plot_type,
    timestep,
    cbar,
    ax
):
    """
    Plot a single sample field with pressure contour and streamlines.
    
    Parameters:
    -----------
    sample_coords : array
        Sample coordinates (x, y)
    sample_u : array
        Velocity in x-direction
    sample_v : array
        Velocity in y-direction
    sample_pressure : array
        Pressure field
    boundary_coords : tuple
        Tuple of boundary coordinate arrays
    mask : array
        Mask for valid regions
    plot_type : str
        Type of plot ('baseline', 'compensated', 'uncompensate')
    timestep : int
        Timestep index
    cbar : bool
        Whether to show colorbars
    ax : matplotlib.axes.Axes
        Axes to plot on
    """
    
    x = sample_coords[:, 0]
    y = sample_coords[:, 1]

    # Create grid for interpolation
    xi = np.linspace(x.min(), x.max(), 200)
    yi = np.linspace(y.min(), y.max(), 200)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate data to regular grid
    ui_sample = griddata((x, y), sample_u, (xi, yi), method='linear')
    vi_sample = griddata((x, y), sample_v, (xi, yi), method='linear')
    pressure_sample = griddata((x, y), sample_pressure, (xi, yi), method='linear')

    # Apply mask
    ui_sample[~mask] = np.nan
    vi_sample[~mask] = np.nan
    pressure_sample[~mask] = np.nan

    # Calculate speed magnitude
    speed_sample = np.sqrt(ui_sample**2 + vi_sample**2)

    # Set plot limits
    ax.set_xlim(-10, 70)
    ax.set_ylim(-10, 70)
    ax.set_aspect('equal')

    # Set colormap and ranges
    custom_cmap = 'RdBu_r'
    pressure_min = -4
    pressure_max = 4
    speed_min = 0
    speed_max = 4

    # Clip pressure values
    pressure_sample = np.clip(pressure_sample, pressure_min, pressure_max)

    # Plot pressure contour
    im1 = ax.contourf(xi, yi, pressure_sample, 
                      levels=np.linspace(pressure_min, pressure_max, 200), 
                      cmap=custom_cmap, alpha=1, vmin=pressure_min, vmax=pressure_max)
    
    # Plot streamlines
    strm1 = ax.streamplot(xi, yi, ui_sample, vi_sample, color=speed_sample, 
                           cmap='viridis', linewidth=0.5, density=0.55, 
                           arrowsize=0.5, broken_streamlines=False)
    
    # Plot boundaries
    for boundary in boundary_coords:
        ax.plot(boundary[:, 0], boundary[:, 1], 'k-', linewidth=0.5)
    
    ax.axis('off')
    ax.set_frame_on(False)
    
    # Add title
    ax.set_title(f'{plot_type}\n Time instant: {timestep}', fontsize=8)
    
    # Add colorbars if requested
    if cbar:
        # Pressure colorbar
        cbar1 = plt.colorbar(im1, ax=ax, label='Pressure', shrink=1)
        cbar1.set_ticks(np.arange(pressure_min, pressure_max+0.1, 1))
        
        # Speed colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis')
        sm.set_clim(vmin=speed_min, vmax=speed_max)
        sm.set_array(speed_sample)
        cbar2 = plt.colorbar(sm, ax=ax, label='Speed', shrink=1)
        cbar2.set_ticks(np.linspace(speed_min, speed_max, 5))
    
    return im1


def main():
    """Main function to create 3x4 grid of plots for all plot types."""
    
    # Load data
    sample_coords = np.load("../coords.npy")
    mask = np.load("../boundary/mask.npy")
    boundary_coords = (
        pd.read_csv("../boundary/boundary-1.csv", header=None).values,
        pd.read_csv("../boundary/boundary-2.csv", header=None).values,
        pd.read_csv("../boundary/boundary-3.csv", header=None).values,
    )
    
    # Define plot types
    plot_types = ['uncompensate', 'compensated', 'baseline']
    
    # Create figure with 3 rows (plot types) and 4 columns (timesteps)
    fig, axes = plt.subplots(3, 4, figsize=(10, 6), dpi=300)
    
    # Plot ID for selecting samples
    plot_id = 12
    
    # Loop through plot types (rows)
    for row_idx, plot_type in enumerate(plot_types):
        # Load data for this plot type
        data = np.load(f"gen_data_{plot_type}.npy")
        
        # Loop through timesteps (columns)
        for col_idx in range(4):
            # Select sample
            sample_idx = col_idx + 4 * plot_id
            sample_u = data[sample_idx, :, 0]
            sample_v = data[sample_idx, :, 1]
            sample_pressure = data[sample_idx, :, 2]
            
            # Plot on corresponding subplot
            ax = axes[row_idx, col_idx]
            plot_sample_field(
                sample_coords=sample_coords,
                sample_u=sample_u,
                sample_v=sample_v,
                sample_pressure=sample_pressure,
                boundary_coords=boundary_coords,
                mask=mask,
                plot_type=plot_type,
                timestep=col_idx,
                cbar=False,
                ax=ax
            )
    
    # Adjust layout
    plt.tight_layout()
    
    # Add overall colorbars
    fig.subplots_adjust(right=0.85)
    
    # Add pressure colorbar
    cbar_ax_pressure = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    sm_pressure = plt.cm.ScalarMappable(cmap='RdBu_r')
    sm_pressure.set_clim(-4, 4)
    cbar_pressure = fig.colorbar(sm_pressure, cax=cbar_ax_pressure)
    cbar_pressure.set_label('Pressure', fontsize=10)
    
    # Add speed colorbar
    cbar_ax_speed = fig.add_axes([0.97, 0.15, 0.02, 0.7])
    sm_speed = plt.cm.ScalarMappable(cmap='viridis')
    sm_speed.set_clim(0, 4)
    cbar_speed = fig.colorbar(sm_speed, cax=cbar_ax_speed)
    cbar_speed.set_label('Velocity', fontsize=10)
    
    # Save and show plot
    plt.savefig('field_comparison.png', bbox_inches='tight', dpi=300)
    # plt.show()


if __name__ == "__main__":
    main()
    print("Plot saved as 'field_comparison.png'")
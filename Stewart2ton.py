import plotly.graph_objects as go
from StewartClass import StewartPlatform

# Define parameters
r_b = 1325  # Radius of base in mm
phi_b = 108  # Angle between base joints
r_p = 732.5  # Radius of platform
phi_p = 14  # Angle between platform joints

# Create Stewart Platform instance
platform = StewartPlatform(r_b, phi_b, r_p, phi_p)

pose = [0, 0, 1450, 0, 0, 0]  # [x, y, z, roll, pitch, yaw]
leg_lengths = platform.getIK(pose)
platform.plot()

# Calculate Platform Forces given Actuator Forces
F_actuators = [100, 100, 100, 200, 100, 30]  # Example actuator forces
F_platform = platform.getPlatformForces(F_actuators)

# Calculate Actuator Forces given Platform Forces
F_platform = [900, 900, 900, 900, 900, 900]  # Example platform forces
F_actuators = platform.getActuatorForces(F_platform)
print(F_actuators)

# Get Force Ellipsoid
force_ellipsoid = platform.getForceEllipsoid()
print("Force Ellipsoid:", force_ellipsoid)


# Get Local Design Index (LDI)
# Local design index for Force transmittability (actuator design)
ldi = platform.getLDI()
print(ldi)

# Define workspace limits [x_min, x_max, y_min, y_max, z_min, z_max]
workspace_limits = [-400, 400, -400, 400, -240, 240]
RPY = [21, 22, 23]  # Fixed orientation (roll, pitch, yaw)
N = 10  # Number of points in each dimension
choice = 4  # Choice of index calculation (1: Singular Value Index, etc.)
# self.options = {
#     1: self.getSingularValueIndex, # measures drive capability of the platform, finds max q_dot under unitary x_dot
#     2: self.getManipulabilityIndex,# Measures manipulability of manipulator, can be used to optimize it's configuration
#     3: self.getConditionNumber,# Measures closeness to isotropic configuration [1,+ inf)
#     4: self.getLocalConditionIndex,# Measures closeness to isotropic configuration (0,1]
#     5: self.getLDI # Local design index for Force transmittability (actuator design)
#     6: self.getLocalConditionIndexT # Measures closeness to force isotropic configuration, 0 when joint forces go to infinity.
# }

workspace_indices_position = platform.getIndexWorkspacePosition(
    workspace_limits, RPY, N, choice)
print("Workspace Indices (Position):", workspace_indices_position)

# Define orientation limits [roll_min, roll_max, pitch_min, pitch_max, yaw_min, yaw_max]
orientation_limits = [-21, 21, -22, 22, -23, 23]
position = [0, 0, 0.4]  # Fixed position

workspace_indices_orientation = platform.getIndexWorkspaceOrientation(
    position, orientation_limits, N, choice)
print("Workspace Indices (Orientation):", workspace_indices_orientation)


# Extract data for plotting
x = [point[0] for point in workspace_indices_orientation]
y = [point[1] for point in workspace_indices_orientation]
z = [point[2] for point in workspace_indices_orientation]
values = [point[3] for point in workspace_indices_orientation]

# Create 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=x, y=y, z=z, mode='markers',
    marker=dict(
        size=5,
        color=values,  # Set color to the index values
        colorscale='Viridis',  # Choose a colorscale
        colorbar=dict(title='Index Value')
    )
)])

# Set plot title and axis labels
fig.update_layout(
    title='Workspace Indices (Orientation)',
    scene=dict(
        xaxis_title='Roll',
        yaxis_title='Pitch',
        zaxis_title='Yaw'
    )
)

# Show plot
fig.show()


# Singularity Finder
# Define workspace limits
workspace_limits = [-0.5, 0.5, -0.5, 0.5, 0.1, 0.6]
orientation_limits = [-10, 10, -10, 10, -10, 10]
# Define number of points for dimension
N_pos = 10  # Number of points in each dimension
N_orient = 10  # Number of points in each dimension

# Choosing N_pos and N_orient too high may result in a computational expensive operation, suggested values ( N_pos=10, N_orient=10 )
# for practical usage there is the need to filter the data. Suggestion: filter by local condition index value AND by distance between data points (from scipy.spatial.distance import cdist).

singularities_task_space = spider.getSingularityWorkspace(
    workspace_limits, orientation_limits, N_pos, N_orient)  # find singularities in all space

print("Singularities in task space:", singularities_task_space)
